#include <Siv3D.hpp> // Siv3D v0.6.16

// ============================================================================
// 1. AudioCore: 音声解析の基盤
//    MFCC抽出などの数学的な処理を担当します.
// ============================================================================

namespace AudioCore
{
	// 音量の RMS 変換ヘルパー.
	[[nodiscard]] double VolumeToRMS(const double volume)
	{
		return Clamp(Math::Pow(10.0, (volume - 1.0) * 5.0), 0.0, 1.0);
	}

	struct MFCC
	{
		Array<double> feature;

		[[nodiscard]] bool isUnset() const
		{
			return std::ranges::all_of(feature, [](const double x)
			{
				return x == 0.0;
			});
		}

		[[nodiscard]] double norm() const
		{
			return Math::Sqrt(std::accumulate(
				feature.begin(), feature.end(), 0.0,
				[](const auto& norm, const auto& x)
				{
					return norm + x * x;
				}));
		}

		[[nodiscard]] double cosineSimilarity(const MFCC& other) const
		{
			if (feature.size() != other.feature.size())
				return 0.0;
			const double thisNorm = norm(), otherNorm = other.norm();
			if (thisNorm < 1e-8 || otherNorm < 1e-8)
				return 0.0;

			double innerProduct = 0.0;
			for (size_t i : step(feature.size()))
				innerProduct += feature[i] * other.feature[i];
			return innerProduct / thisNorm / otherNorm;
		}

		// 複数の特徴量の平均を算出する.
		static MFCC Average(const Array<double>& sumBuffer, int32 count)
		{
			if (count <= 0 || sumBuffer.isEmpty())
				return MFCC{};
			return MFCC{sumBuffer.map([&](double x)
			{
				return x / count;
			})};
		}
	};

	class MFCCAnalyzer
	{
	public:
		static [[nodiscard]] double freqToMel(const double freq)
		{
			return 1127.01 * Math::Log(1.0 + freq / 700.0);
		}

		static [[nodiscard]] double melToFreq(const double mel)
		{
			return 700.0 * (Math::Exp(mel / 1127.01) - 1.0);
		}

		explicit MFCCAnalyzer(const FFTSampleLength frames = FFTSampleLength::SL2K, const size_t melChannels = 40, const size_t mfccOrder = 12)
			: frames(frames), f(256uLL << FromEnum(frames), 0.0f), melChannels(melChannels), bin(melChannels + 2),
			  melSpectrum(melChannels), melEnvelope(melChannels), mfccOrder(mfccOrder)
		{
		}

		[[nodiscard]] MFCC analyze(const Microphone& mic)
		{
			if (not mic.isLoop())
				throw Error{U"Microphone is must be loop mode."};
			if (not mic.isRecording() || mic.getBufferLength() < f.size())
				return MFCC{Array<double>(mfccOrder, 0.0)};

			const auto sampleRate = mic.getSampleRate();
			const auto& buffer = mic.getBuffer();
			const size_t writePos = mic.posSample();

			// バッファのコピー.
			for (size_t pos : step(f.size()))
			{
				const size_t idx = (pos + writePos < f.size() ? mic.getBufferLength() : 0) + pos + writePos - f.size();
				f[pos] = buffer[idx].left;
			}

			// プリエンファシス.
			for (size_t i : Range(f.size() - 1, 1, -1))
			{
				f[i] -= f[i - 1] * 0.96875f;
			}

			// ハミング窓.
			for (size_t i : Range(f.size() - 2, 1))
			{
				f[i] *= static_cast<float>(0.54 - 0.46 * cos(2.0 * Math::Pi * i / (f.size() - 1)));
			}
			f.front() = 0.0f;
			f.back() = 0.0f;

			FFT::Analyze(fftResult, f.data(), f.size(), sampleRate, frames);

			const auto melMax = freqToMel(static_cast<double>(sampleRate) / 2.0);
			const auto melMin = freqToMel(0);
			const auto deltaMel = (melMax - melMin) / static_cast<double>(melChannels + 1);

			// bin 計算.
			for (size_t i : step(bin.size()))
			{
				bin[i] = static_cast<size_t>((f.size() + 1) * melToFreq(melMin + i * deltaMel) / sampleRate);
			}

			for (size_t i : step(melChannels))
			{
				melSpectrum[i] = 0.0;
				for (size_t j : Range(bin[i], bin[i + 1] - 1))
				{
					melSpectrum[i] += static_cast<double>(fftResult.buffer[j]) * (j - bin[i]) / (bin[i + 1] - bin[i]);
				}
				for (size_t j : Range(bin[i + 1], bin[i + 2] - 1))
				{
					melSpectrum[i] += static_cast<double>(fftResult.buffer[j]) * (bin[i + 2] - j) / (bin[i + 2] - bin[i + 1]);
				}
				melEnvelope[i] = {2.0 * bin[i + 1] / f.size(), melSpectrum[i] / (bin[i + 2] - bin[i])};
			}

			MFCC mfcc{Array<double>(mfccOrder, 0.0)};
			for (size_t i : Range(1, mfccOrder))
			{
				for (size_t j : step(melChannels))
				{
					mfcc.feature[i - 1] += Math::Log10(Math::Abs(melSpectrum[j])) * Math::Cos(Math::Pi * i * (j + 0.5) / melChannels) * 10;
				}
			}
			return mfcc;
		}

	protected:
		FFTSampleLength frames;
		Array<float> f;
		FFTResult fftResult;
		size_t melChannels;
		Array<size_t> bin;
		Array<double> melSpectrum;
		Array<Vec2> melEnvelope;
		size_t mfccOrder;
	};
} // namespace AudioCore

// ============================================================================
// 2. GameSystem: ゲームロジック
//    プレイヤー制御や、音声コマンドの管理 (学習・判定) を担当します.
// ============================================================================

namespace GameSystem
{
	using namespace AudioCore;

	// 定数パラメータ.
	struct Config
	{
		static constexpr double InputVolumeThreshold = 0.5; // 入力とみなす音量 (0.0 - 1.0).
		static constexpr double SimilarityThreshold = 0.85; // コサイン類似度の判定閾値.
		static constexpr int32 StabilityFrames = 5; // チャタリング対策の安定化フレーム数.
		static constexpr double PlayerSpeed = 200.0; // プレイヤーの移動速度.
	};

	// プレイヤーの実体.
	class Player
	{
	public:
		Player(const Vec2& initialPos): m_pos(initialPos)
		{
		}

		void update(const String& command, double deltaTime)
		{
			if (command == U"あ")
			{
				m_color = Palette::Red;
			}
			else
			{
				m_color = Palette::Skyblue;

				if (command == U"い")
					m_pos.x -= Config::PlayerSpeed * deltaTime;
				else if (command == U"う")
					m_pos.y -= Config::PlayerSpeed * deltaTime;
				else if (command == U"え")
					m_pos.x += Config::PlayerSpeed * deltaTime;
				else if (command == U"お")
					m_pos.y += Config::PlayerSpeed * deltaTime;
			}

			// 画面内への制限.
			m_pos = m_pos.clamp(Scene::Rect().stretched(-20));
		}

		void draw() const
		{
			Circle{m_pos, 30}.draw(m_color);
		}

	private:
		Vec2 m_pos;
		Color m_color = Palette::Skyblue;
	};

	// 学習データの1スロット.
	struct LearningSlot
	{
		String label;
		MFCC mfcc;
		bool isRecorded = false;
	};

	// 音声コマンドの管理システム (学習・判定・安定化).
	class VoiceCommandSystem
	{
	public:
		VoiceCommandSystem()
		{
			m_slots = {
				{U"あ", {}, false}, {U"い", {}, false}, {U"う", {}, false}, {U"え", {}, false}, {U"お", {}, false}};
		}

		// マイク入力からコマンドを判定する (ゲームフェーズ用).
		String detectCommand(const MFCC& inputMFCC, double inputRMS)
		{
			// 音量が小さい、またはMFCCが無効なら即座に停止処理.
			if (inputRMS <= VolumeToRMS(Config::InputVolumeThreshold) || inputMFCC.isUnset())
			{
				// ステートを全てリセットして「無」を返す.
				m_potentialVowel = U"";
				m_stabilityCount = 0;
				m_confirmedVowel = U""; // これを空にすることで、キャラが止まります.
				return m_confirmedVowel;
			}

			// 最も近い母音を探す.
			String bestLabel = U"";
			double maxSimilarity = 0.0;
			int bestIndex = -1;

			for (size_t i : step(m_slots.size()))
			{
				double similarity = inputMFCC.cosineSimilarity(m_slots[i].mfcc);
				if (similarity > maxSimilarity)
				{
					maxSimilarity = similarity;
					bestIndex = static_cast<int>(i);
				}
			}

			if (bestIndex != -1 && maxSimilarity > Config::SimilarityThreshold)
			{
				bestLabel = m_slots[bestIndex].label;
			}

			// チャタリング対策を通す.
			updateStability(bestLabel);
			return m_confirmedVowel;
		}

		// 学習用: 特徴量を累積する.
		void accumulateForLearning(const MFCC& mfcc)
		{
			if (m_learningBufferSum.isEmpty())
			{
				m_learningBufferSum.resize(mfcc.feature.size(), 0.0);
			}
			for (size_t i : step(m_learningBufferSum.size()))
			{
				m_learningBufferSum[i] += mfcc.feature[i];
			}
			m_learningSampleCount++;
		}

		// 学習用: 累積をリセットする.
		void resetLearningBuffer()
		{
			m_learningBufferSum.clear();
			m_learningSampleCount = 0;
		}

		// 学習用: 現在の累積結果をスロットに保存する.
		bool commitLearning(int32 slotIndex)
		{
			if (slotIndex < 0 || slotIndex >= (int32) m_slots.size())
				return false;
			if (m_learningSampleCount <= 10)
				return false; // サンプル不足は無視.

			m_slots[slotIndex].mfcc = MFCC::Average(m_learningBufferSum, m_learningSampleCount);
			m_slots[slotIndex].isRecorded = true;
			return true;
		}

		// データアクセサ.
		int32 getLearningSampleCount() const { return m_learningSampleCount; }
		Array<LearningSlot>& getSlots() { return m_slots; }
		const Array<LearningSlot>& getSlots() const { return m_slots; }

		bool isAllRecorded() const
		{
			return std::all_of(m_slots.begin(), m_slots.end(), [](const auto& s)
			{
				return s.isRecorded;
			});
		}

		String getPotentialVowel() const { return m_potentialVowel; } // 判定中(未確定)の母音.

	private:
		Array<LearningSlot> m_slots;

		// 学習用の一時バッファ.
		Array<double> m_learningBufferSum;
		int32 m_learningSampleCount = 0;

		// チャタリング対策用ステート.
		String m_potentialVowel = U""; // 現在候補となっている母音.
		String m_confirmedVowel = U""; // 確定した母音.
		int32 m_stabilityCount = 0;

		void updateStability(const String& currentBest)
		{
			if (currentBest != U"" && currentBest == m_potentialVowel)
			{
				m_stabilityCount++;
			}
			else
			{
				m_potentialVowel = currentBest;
				m_stabilityCount = 0;
			}

			if (m_stabilityCount > Config::StabilityFrames)
			{
				m_confirmedVowel = m_potentialVowel;
			}
		}
	};
} // namespace GameSystem

// ============================================================================
// 3. UserInterface: 画面UI制御
//    学習画面とゲーム画面の具体的な描画・入力フローを管理します.
// ============================================================================

namespace UserInterface
{
	using namespace AudioCore;
	using namespace GameSystem;

	class AppUI
	{
	public:
		AppUI()
			: m_player(Scene::Center()), m_font(40), m_smallFont(20)
		{
		}

		// アプリケーション全体の更新ループ.
		void update(const Microphone& mic, MFCCAnalyzer& analyzer)
		{
			const MFCC mfcc = analyzer.analyze(mic);
			const double rms = mic.rootMeanSquare();

			if (m_isGameMode)
			{
				updateGamePhase(mfcc, rms);
			}
			else
			{
				updateLearningPhase(mfcc, rms);
			}
		}

	private:
		// ステート.
		bool m_isGameMode = false;
		int32 m_selectedSlotIndex = 0;

		// リソース & サブシステム.
		Player m_player;
		VoiceCommandSystem m_voiceSystem;
		Font m_font;
		Font m_smallFont;

		// 学習フェーズのロジック.
		void updateLearningPhase(const MFCC& mfcc, double rms)
		{
			Scene::SetBackground(Palette::Darkgray);
			m_font(U"学習モード: 選択してSPACEキーを長押し").drawAt(400, 50, Palette::White);

			auto& slots = m_voiceSystem.getSlots();

			// キー操作でのスロット移動.
			if (KeyRight.down())
				m_selectedSlotIndex = (m_selectedSlotIndex + 1) % slots.size();
			if (KeyLeft.down())
				m_selectedSlotIndex = (m_selectedSlotIndex + (int32) slots.size() - 1) % slots.size();

			// UI レイアウト定数.
			const int32 boxSize = 100;
			const int32 gap = 20;
			const int32 startX = (Scene::Width() - (boxSize * 5 + gap * 4)) / 2;
			const int32 startY = 200;

			// スロット描画.
			for (int32 i : step(slots.size()))
			{
				const Rect box{startX + i * (boxSize + gap), startY, boxSize, boxSize};
				const bool isSelected = (i == m_selectedSlotIndex);
				const bool isRecordingInput = KeySpace.pressed() && rms > VolumeToRMS(Config::InputVolumeThreshold);
				const bool isRecordingNow = isSelected && isRecordingInput;

				// マウス選択.
				if (box.leftClicked())
					m_selectedSlotIndex = i;

				// 描画: 背景.
				box.rounded(10).draw(isSelected ? ColorF{0.3, 0.3, 0.4} : ColorF{0.2});
				if (isSelected)
					box.rounded(10).drawFrame(4, Palette::Skyblue);

				// 描画: ラベル.
				m_font(slots[i].label).drawAt(box.center(), Palette::White);

				// 描画: 完了マーク.
				if (slots[i].isRecorded)
					m_smallFont(U"OK").drawAt(box.bottomCenter().movedBy(0, -20), Palette::Lightgreen);

				// 描画: 録音進捗アニメーション.
				if (isRecordingNow)
				{
					const double progress = Min(m_voiceSystem.getLearningSampleCount() / 60.0, 1.0);
					Circle{box.center(), 40}.drawArc(0_deg, 360_deg * progress, 4, 0, Palette::Orange);
				}
			}

			// 録音ロジック制御.
			handleRecordingLogic(mfcc, rms);

			// ゲーム開始ボタン制御.
			if (m_voiceSystem.isAllRecorded())
			{
				const Rect startBtn = Rect{Arg::center(400, 450), 300, 60};
				startBtn.rounded(10).draw(startBtn.mouseOver() ? Palette::Orange : Palette::Darkorange);
				m_font(U"ゲーム開始").drawAt(startBtn.center(), Palette::White);

				if (startBtn.leftClicked() || KeyEnter.down())
					m_isGameMode = true;
			}
			else
			{
				m_smallFont(U"すべての音声を登録してください").drawAt(400, 450, Palette::Gray);
			}
		}

		void handleRecordingLogic(const MFCC& mfcc, double rms)
		{
			// スペースキー押下開始: リセット.
			if (KeySpace.down())
			{
				m_voiceSystem.resetLearningBuffer();
			}

			// 押下中: 累積.
			if (KeySpace.pressed() && rms > VolumeToRMS(Config::InputVolumeThreshold) && !mfcc.isUnset())
			{
				m_voiceSystem.accumulateForLearning(mfcc);
			}

			// 離した瞬間: 保存処理.
			if (KeySpace.up())
			{
				bool success = m_voiceSystem.commitLearning(m_selectedSlotIndex);

				if (success)
				{
					// 自動で次の未録音スロットへ移動.
					const auto& slots = m_voiceSystem.getSlots();
					for (int32 k = 1; k < (int32) slots.size(); ++k)
					{
						int32 nextIdx = (m_selectedSlotIndex + k) % slots.size();
						if (!slots[nextIdx].isRecorded)
						{
							m_selectedSlotIndex = nextIdx;
							break;
						}
					}
				}
				m_voiceSystem.resetLearningBuffer();
			}
		}

		// ゲームフェーズのロジック.
		void updateGamePhase(const MFCC& mfcc, double rms)
		{
			Scene::SetBackground(Palette::White);
			m_font(U"声で操作してください").drawAt(400, 50, Palette::Black);

			// コマンド判定.
			String command = m_voiceSystem.detectCommand(mfcc, rms);
			String potential = m_voiceSystem.getPotentialVowel();

			// プレイヤー更新・描画.
			m_player.update(command, Scene::DeltaTime());
			m_player.draw();

			// デバッグ表示.
			if (command)
				m_font(command).drawAt(Scene::Center().movedBy(0, -100), Palette::Black);
			else if (potential)
				m_font(potential).drawAt(Scene::Center().movedBy(0, -100), ColorF{0.8});

			// ガイド表示.
			m_font(U"あ:赤  い:←  う:↑  え:→  お:↓").drawAt(400, 550, Palette::Gray);

			// 再学習ボタン.
			if (SimpleGUI::Button(U"再学習", Vec2{20, 20}))
			{
				m_isGameMode = false;
			}
		}
	};
} // namespace UserInterface

// ============================================================================
// Main: エントリーポイント
// ============================================================================

void Main()
{
	Window::SetTitle(U"Voice Controller Game");
	Window::Resize(800, 600);

	// 音声解析エンジンの初期化.
	AudioCore::MFCCAnalyzer mfccAnalyzer{};
	Microphone mic{StartImmediately::Yes};

	// UI & アプリケーション管理.
	UserInterface::AppUI appUI;

	while (System::Update())
	{
		// マイクの再接続チェック.
		if (System::EnumerateMicrophones().none([&](const auto& info)
		{
			return info.microphoneIndex == mic.microphoneIndex();
		}))
		{
			mic.open(StartImmediately::Yes);
		}

		// アプリケーション更新.
		appUI.update(mic, mfccAnalyzer);
	}
}
