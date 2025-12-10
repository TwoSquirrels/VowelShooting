#include <Siv3D.hpp> // Siv3D v0.6.16

// --- MFCC 関連クラス群 (変更なし) ---

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

		for (size_t pos : step(f.size()))
		{
			const size_t idx = (pos + writePos < f.size() ? mic.getBufferLength() : 0) + pos + writePos - f.size();
			f[pos] = buffer[idx].left;
		}

		for (size_t i : Range(f.size() - 1, 1, -1))
			f[i] -= f[i - 1] * 0.96875f;

		for (size_t i : Range(f.size() - 2, 1))
			f[i] *= static_cast<float>(0.54 - 0.46 * cos(2.0 * Math::Pi * i / (f.size() - 1)));
		f.front() = 0.0f;
		f.back() = 0.0f;

		FFT::Analyze(fftResult, f.data(), f.size(), sampleRate, frames);

		const auto melMax = freqToMel(static_cast<double>(sampleRate) / 2.0);
		const auto melMin = freqToMel(0);
		const auto deltaMel = (melMax - melMin) / static_cast<double>(melChannels + 1);

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

// --- ゲーム関連構造体 ---

[[nodiscard]] double volumeToRMS(const double volume)
{
	return Clamp(Math::Pow(10.0, (volume - 1.0) * 5.0), 0.0, 1.0);
}

struct Player
{
	Vec2 pos;
	Color color;
	double speed = 200.0;

	void update(const String& command, double deltaTime)
	{
		if (command == U"あ")
			color = Palette::Red;
		else
		{
			color = Palette::Skyblue;
			if (command == U"い")
				pos.x -= speed * deltaTime;
			else if (command == U"う")
				pos.y -= speed * deltaTime;
			else if (command == U"え")
				pos.x += speed * deltaTime;
			else if (command == U"お")
				pos.y += speed * deltaTime;
		}
		pos = pos.clamp(Scene::Rect().stretched(-20));
	}

	void draw() const
	{
		Circle{pos, 30}.draw(color);
	}
};

// 学習データのスロット管理用.
struct LearningSlot
{
	String label;
	MFCC mfcc;
	bool isRecorded = false;
};

void Main()
{
	Window::SetTitle(U"Voice Controller Game");
	Window::Resize(800, 600);

	MFCCAnalyzer mfccAnalyzer{};
	Microphone mic{StartImmediately::Yes};

	const double rmsThreshold = volumeToRMS(0.5);

	// 学習データの初期化.
	Array<LearningSlot> slots = {
		{U"あ", {}, false},
		{U"い", {}, false},
		{U"う", {}, false},
		{U"え", {}, false},
		{U"お", {}, false}};

	int32 selectedSlotIndex = 0; // 現在選択中のスロット.
	bool isGameMode = false;

	// チャタリング対策用変数.
	String potentialVowel = U"";
	int32 stabilityCount = 0;
	const int32 STABILITY_THRESHOLD = 5;

	// 平均化学習用変数.
	Array<double> currentFeatureSum;
	int32 currentSampleCount = 0;

	Player player{Scene::Center(), Palette::Skyblue};
	Font font{40};
	Font smallFont{20};

	while (System::Update())
	{
		if (System::EnumerateMicrophones().none([&](const auto& info)
		{
			return info.microphoneIndex == mic.microphoneIndex();
		}))
		{
			mic.open(StartImmediately::Yes);
		}

		const auto mfcc = mfccAnalyzer.analyze(mic);
		const double currentRMS = mic.rootMeanSquare();

		if (not isGameMode)
		{
			// --- 学習フェーズ (UI刷新) ---
			Scene::SetBackground(Palette::Darkgray);

			font(U"学習モード: 選択してSPACEキーを長押し").drawAt(400, 50, Palette::White);

			// キー操作でのスロット移動.
			if (KeyRight.down())
				selectedSlotIndex = (selectedSlotIndex + 1) % slots.size();
			if (KeyLeft.down())
				selectedSlotIndex = (selectedSlotIndex + (int32) slots.size() - 1) % slots.size();

			// スロットの描画とロジック.
			const int32 boxSize = 100;
			const int32 gap = 20;
			const int32 startX = (Scene::Width() - (boxSize * 5 + gap * 4)) / 2;
			const int32 startY = 200;

			for (int32 i : step(slots.size()))
			{
				const Rect box{startX + i * (boxSize + gap), startY, boxSize, boxSize};
				const bool isSelected = (i == selectedSlotIndex);
				const bool isRecordingNow = isSelected && KeySpace.pressed() && currentRMS > rmsThreshold;

				// マウス選択処理.
				if (box.leftClicked())
				{
					selectedSlotIndex = i;
				}

				// 背景描画.
				box.rounded(10).draw(isSelected ? ColorF{0.3, 0.3, 0.4} : ColorF{0.2});
				if (isSelected)
				{
					box.rounded(10).drawFrame(4, Palette::Skyblue);
				}

				// 文字描画.
				font(slots[i].label).drawAt(box.center(), Palette::White);

				// 状態表示 (完了マーク).
				if (slots[i].isRecorded)
				{
					smallFont(U"OK").drawAt(box.bottomCenter().movedBy(0, -20), Palette::Lightgreen);
				}

				// 録音中の演出 (進捗円).
				if (isRecordingNow)
				{
					const double progress = Min(currentSampleCount / 60.0, 1.0);
					Circle{box.center(), 40}.drawArc(0_deg, 360_deg * progress, 4, 0, Palette::Orange);
				}
			}

			// --- 学習ロジック (選択中のスロットに対して実行) ---
			if (KeySpace.down())
			{
				currentFeatureSum.clear();
				currentSampleCount = 0;
			}

			if (KeySpace.pressed() && currentRMS > rmsThreshold && !mfcc.isUnset())
			{
				if (currentFeatureSum.isEmpty())
					currentFeatureSum.resize(mfcc.feature.size(), 0.0);

				for (size_t k : step(currentFeatureSum.size()))
				{
					currentFeatureSum[k] += mfcc.feature[k];
				}
				currentSampleCount++;
			}

			if (KeySpace.up())
			{
				if (currentSampleCount > 10) // ある程度サンプルがないと無効とする.
				{
					MFCC averagedMFCC;
					averagedMFCC.feature = currentFeatureSum.map([&](double x)
					{
						return x / currentSampleCount;
					});

					// 選択中のスロットに保存.
					slots[selectedSlotIndex].mfcc = averagedMFCC;
					slots[selectedSlotIndex].isRecorded = true;

					// 自動で次の未録音スロットへ移動 (便利機能).
					for (int32 k = 1; k < (int32) slots.size(); ++k)
					{
						int32 nextIdx = (selectedSlotIndex + k) % slots.size();
						if (!slots[nextIdx].isRecorded)
						{
							selectedSlotIndex = nextIdx;
							break;
						}
					}
				}
				currentFeatureSum.clear();
				currentSampleCount = 0;
			}

			// 全て完了しているかチェック.
			const bool allRecorded = std::all_of(slots.begin(), slots.end(), [](const auto& s)
			{
				return s.isRecorded;
			});

			if (allRecorded)
			{
				const Rect startBtn = Rect{Arg::center(400, 450), 300, 60};
				startBtn.rounded(10).draw(startBtn.mouseOver() ? Palette::Orange : Palette::Darkorange);
				font(U"ゲーム開始").drawAt(startBtn.center(), Palette::White);

				if (startBtn.leftClicked() || KeyEnter.down())
				{
					isGameMode = true;
				}
			}
			else
			{
				smallFont(U"すべての音声を登録してください").drawAt(400, 450, Palette::Gray);
			}
		}
		else
		{
			// --- ゲームフェーズ ---
			Scene::SetBackground(Palette::White);
			font(U"声で操作してください").drawAt(400, 50, Palette::Black);

			String confirmedVowel = U"";
			String currentBestVowel = U"";
			double maxSimilarity = 0.0;

			if (currentRMS > rmsThreshold && !mfcc.isUnset())
			{
				int bestIndex = -1;

				for (size_t i : step(slots.size()))
				{
					double similarity = mfcc.cosineSimilarity(slots[i].mfcc);
					if (similarity > maxSimilarity)
					{
						maxSimilarity = similarity;
						bestIndex = static_cast<int>(i);
					}
				}

				if (bestIndex != -1 && maxSimilarity > 0.85)
				{
					currentBestVowel = slots[bestIndex].label;
				}
			}

			// チャタリング対策.
			if (currentBestVowel != U"" && currentBestVowel == potentialVowel)
			{
				stabilityCount++;
			}
			else
			{
				potentialVowel = currentBestVowel;
				stabilityCount = 0;
			}

			if (stabilityCount > STABILITY_THRESHOLD)
			{
				confirmedVowel = potentialVowel;
			}

			player.update(confirmedVowel, Scene::DeltaTime());
			player.draw();

			if (confirmedVowel)
			{
				font(confirmedVowel).drawAt(Scene::Center().movedBy(0, -100), Palette::Black);
			}
			else if (potentialVowel)
			{
				font(potentialVowel).drawAt(Scene::Center().movedBy(0, -100), ColorF{0.8});
			}

			// ガイド表示.
			font(U"あ:赤  い:←  う:↑  え:→  お:↓").drawAt(400, 550, Palette::Gray);

			// 再学習に戻るボタン.
			if (SimpleGUI::Button(U"再学習", Vec2{20, 20}))
			{
				isGameMode = false;
			}
		}
	}
}
