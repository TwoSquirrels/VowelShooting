#include <Siv3D.hpp> // Siv3D v0.6.16

// ============================================================================
// 1. AudioCore: 音声解析の基盤
//    MFCC抽出などの数学的な処理を担当します. (変更なし)
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

		// ユークリッド距離の二乗を計算する (k-NN用).
		[[nodiscard]] double distSq(const MFCC& other) const
		{
			if (feature.size() != other.feature.size())
				return DBL_MAX;

			double sum = 0.0;
			for (size_t i : step(feature.size()))
			{
				const double d = feature[i] - other.feature[i];
				sum += d * d;
			}
			return sum;
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
} // namespace AudioCore

// ============================================================================
// 2. DanmakuCore: 弾幕ロジック
//    サンプルの弾幕システムを移植・調整したものです. (変更なし)
// ============================================================================

namespace DanmakuCore
{
	bool isOutOfSceneArea(const Vec2& position)
	{
		constexpr int margin = 20;
		return position.x < -margin || position.x > Scene::Width() + margin || position.y < -margin || position.y > Scene::Height() + margin;
	}

	struct EnemyBullet
	{
		EnemyBullet(const Vec2& _pos, const Vec2& _vel, const Vec2& _acc, float _size)
			: pos(_pos), vel(_vel), acc(_acc), size(_size), stopwatch(StartImmediately::Yes)
		{
		}

		Vec2 pos;
		Vec2 vel;
		Vec2 acc;
		double size;
		Stopwatch stopwatch;

		void update()
		{
			vel += acc * Scene::DeltaTime();
			pos += vel * Scene::DeltaTime();
		}
	};

	class BulletCurtain
	{
	public:
		BulletCurtain()
		{
			mWholePeriod = 10000;
			mBulletMap.emplace(eSpin, Array<EnemyBullet>());
			mBulletMap.emplace(eTail, Array<EnemyBullet>());
			mBulletMap.emplace(eSnow, Array<EnemyBullet>());
			mPrevEnemyPos = Vec2::Zero();
		}

		void clear()
		{
			mStopWatch.reset();
			for (auto& bullets : mBulletMap)
				bullets.second.clear();
			mPrevEnemyPos = Vec2::Zero();
		}

		void start() { mStopWatch.start(); }
		void pause() { mStopWatch.pause(); }

		void update(const Vec2& enemyPos)
		{
			if (mStopWatch.isPaused())
				return;

			updateEvents(enemyPos);
			updateBullets(enemyPos);
			eraseBullets();

			if (mStopWatch.ms() >= mWholePeriod)
			{
				mStopWatch.reset();
				mStopWatch.start();
			}

			mPrevEnemyPos = enemyPos;
		}

		void draw() const
		{
			for (const auto& b : mBulletMap.at(eSnow))
				Circle{b.pos, b.size}.draw(Palette::White);
			for (const auto& b : mBulletMap.at(eTail))
				Circle{b.pos, b.size}.draw(Palette::Hotpink);
			for (const auto& b : mBulletMap.at(eSpin))
				Circle{b.pos, b.size}.draw(Palette::White);
		}

		bool checkHit(const Vec2& pos, const double size)
		{
			for (const auto& enemyBullets : mBulletMap)
				for (const auto& enemyBullet : enemyBullets.second)
					if (enemyBullet.pos.distanceFrom(pos) <= enemyBullet.size + size)
						return true;
			return false;
		}

	private:
		enum BulletKey
		{
			eSpin = 0,
			eTail = 1,
			eSnow = 2
		};

		HashTable<int, Array<EnemyBullet>> mBulletMap;
		Stopwatch mStopWatch;
		int32 mWholePeriod;
		Vec2 mPrevEnemyPos;

		bool triggerMs(int32 triggerTimePoint)
		{
			return abs(mStopWatch.ms() - triggerTimePoint) <= Scene::DeltaTime() * 1000.0 / 1.7;
		}

		bool periodMs(int32 period)
		{
			const int32 now = mStopWatch.ms();
			const double deltams = Scene::DeltaTime() * 1000.0 / 1.5;
			return abs(now % period - period) <= deltams || now % period <= deltams;
		}

		bool passedMs(int32 timePoint) { return mStopWatch.ms() >= timePoint; }

		void updateEvents(const Vec2& enemyPos)
		{
			if (triggerMs(500))
			{
				constexpr int perNum = 3; // 5 -> 3
				for (int i = 0; i < perNum; ++i)
				{
					const double angle = 2 * Math::Pi / perNum * i;
					mBulletMap[eSpin].emplace_back(EnemyBullet(enemyPos, 80.f * Vec2(cos(angle), sin(angle)), Vec2::Zero(), 10)); // 120.f -> 80.f
				}
			}

			if (passedMs(700) && !passedMs(9000) && periodMs(350)) // 250 -> 350
			{
				constexpr double speed = 1.0; // 1.35 -> 1.0
				for (const auto& spinBullet : mBulletMap[eSpin])
				{
					mBulletMap[eTail].emplace_back(EnemyBullet(spinBullet.pos, -speed * spinBullet.vel.rotated(-Math::Pi / 2), Vec2::Zero(), 7));
				}
			}

			if (periodMs(2000 + RandomInt32() % 300)) // 1500 + RandomInt32() % 200 -> 2000 + RandomInt32() % 300
			{
				constexpr int perNum = 2; // 3 -> 2
				for (int i = 0; i < perNum; ++i)
				{
					const double genPos = Random() * 500.0 - 250.0;
					const double speed = Random() * 10.0 + 6.0; // Random() * 15.0 + 10.0 -> Random() * 10.0 + 6.0
					mBulletMap[eSnow].emplace_back(EnemyBullet(Vec2(genPos, -genPos), Vec2::Zero(), Vec2(speed, 1.2 * speed), 5));
				}
			}

			if (triggerMs(9000))
			{
				for (auto& spinBullet : mBulletMap[eSpin])
					spinBullet.acc = 2.0 * spinBullet.vel; // 3.0 -> 2.0
			}
		}

		void updateBullets(const Vec2& enemyPos)
		{
			const Vec2 enemyVelocity = enemyPos - mPrevEnemyPos;

			for (auto& b : mBulletMap[eSpin])
			{
				b.vel.rotate(Math::Pi / 150);
				b.update();
				b.pos += enemyVelocity;
			}
			for (auto& b : mBulletMap[eTail])
			{
				if (b.stopwatch.ms() < 3000)
					b.vel.rotate(Math::Pi / 270);
				else
					b.vel.rotate(Math::Pi / 450);
				b.update();
			}
			for (auto& b : mBulletMap[eSnow])
			{
				b.vel.rotate(-Math::Pi / 2200);
				b.update();
			}
		}

		void eraseBullets()
		{
			mBulletMap.at(eSpin).remove_if([](const EnemyBullet& b)
			{
				return (isOutOfSceneArea(b.pos));
			});
			mBulletMap.at(eTail).remove_if([](const EnemyBullet& b)
			{
				return (isOutOfSceneArea(b.pos));
			});
			mBulletMap.at(eSnow).remove_if([](const EnemyBullet& b)
			{
				constexpr int margin = 20;
				return b.pos.x > Scene::Width() + margin || b.pos.y > Scene::Height() + margin;
			});
		}
	};
} // namespace DanmakuCore

// ============================================================================
// 3. GameSystem: ゲームロジック
//    プレイヤー制御や、音声コマンドの管理 (学習・判定) を担当します.
// ============================================================================

namespace GameSystem
{
	using namespace AudioCore;

	struct Config
	{
		static constexpr double InputVolumeThreshold = 0.1; // 感度を上げるために閾値を低く設定(0.5 -> 0.1).
		static constexpr int32 K_Nearest = 7; // k-NNの k の値.
		static constexpr int32 StabilityFrames = 5;
		static constexpr double PlayerSpeed = 250.0; // 150.0 -> 250.0
		static constexpr double ShotSpeed = 500.0; // 800.0 -> 500.0
		static constexpr double ShotCoolTime = 0.15;
	};

	struct PlayerBullet
	{
		Vec2 pos;
		Vec2 vel;
	};

	class Player
	{
	public:
		Player(const Vec2& initialPos): m_pos(initialPos)
		{
		}

		void update(const String& command, double deltaTime)
		{
			if (command == U"い")
				m_pos.x -= Config::PlayerSpeed * deltaTime;
			else if (command == U"う")
				m_pos.y -= Config::PlayerSpeed * deltaTime;
			else if (command == U"え")
				m_pos.x += Config::PlayerSpeed * deltaTime;
			else if (command == U"お")
				m_pos.y += Config::PlayerSpeed * deltaTime;
			// command == U"雑音" の場合は何も起きない(停止)

			m_pos = m_pos.clamp(Scene::Rect());

			m_shotTimer += deltaTime;
			if (command == U"あ" && m_shotTimer >= Config::ShotCoolTime)
			{
				m_bullets.emplace_back(PlayerBullet{m_pos, Vec2{0, -Config::ShotSpeed}});
				m_shotTimer = 0.0;
			}

			for (auto& b : m_bullets)
				b.pos += b.vel * deltaTime;
			m_bullets.remove_if([](const PlayerBullet& b)
			{
				return b.pos.y < -50;
			});
		}

		void draw() const
		{
			Circle{m_pos, 15}.draw(ColorF{0.25, 0.25, 0.28});

			for (const auto& b : m_bullets)
			{
				Circle{b.pos, 8}.draw(Palette::Orange);
				Circle{b.pos, 5}.draw(Palette::Yellow);
			}
		}

		void reset(const Vec2& pos)
		{
			m_pos = pos;
			m_bullets.clear();
		}

		const Vec2& getPos() const { return m_pos; }
		const Array<PlayerBullet>& getBullets() const { return m_bullets; }

		void removeBullet(size_t index)
		{
			if (index < m_bullets.size())
				m_bullets.remove_if([&](const PlayerBullet& b)
				{
					return &b == &m_bullets[index];
				});
		}

	private:
		Vec2 m_pos;
		Array<PlayerBullet> m_bullets;
		double m_shotTimer = 0.0;
	};

	// 学習データの1スロット.
	struct LearningSlot
	{
		String label;
		Array<MFCC> samples;
		bool isRecorded = false;
	};

	// 音声コマンドの管理システム (k-NN).
	class VoiceCommandSystem
	{
	public:
		VoiceCommandSystem()
		{
			// "雑音" スロットを一番左に配置.
			m_slots = {
				{U"雑音", {}, false}, {U"あ", {}, false}, {U"い", {}, false}, {U"う", {}, false}, {U"え", {}, false}, {U"お", {}, false}};
		}

		String detectCommand(const MFCC& inputMFCC, double inputRMS)
		{
			// 音量による足切りは極小値(無音に近い場合)のみに行う.
			// 0.5 などの高い閾値で弾くと、雑音クラスが判定される前に「判定不能」になってしまうため.
			// ただし MFCC が Unset の場合は判定しようがないので弾く.
			if (inputMFCC.isUnset())
			{
				m_potentialVowel = U"";
				m_stabilityCount = 0;
				m_confirmedVowel = U"";
				return m_confirmedVowel;
			}

			// k-NN アルゴリズム.
			struct Neighbor
			{
				double distSq;
				int32 slotIndex;
			};
			Array<Neighbor> neighbors;

			for (int32 i : step(m_slots.size()))
			{
				for (const auto& sample : m_slots[i].samples)
				{
					double d = inputMFCC.distSq(sample);
					neighbors.push_back({d, i});
				}
			}

			String bestLabel = U"";

			if (not neighbors.isEmpty())
			{
				size_t k = Min<size_t>(Config::K_Nearest, neighbors.size());
				std::partial_sort(neighbors.begin(), neighbors.begin() + k, neighbors.end(),
				                  [](const Neighbor& a, const Neighbor& b)
				                  {
					                  return a.distSq < b.distSq;
				                  });

				HashTable<int32, int32> votes;
				for (size_t i : step(k))
				{
					votes[neighbors[i].slotIndex]++;
				}

				int32 bestSlotIndex = -1;
				int32 maxVotes = -1;
				for (auto [slotIndex, count] : votes)
				{
					if (count > maxVotes)
					{
						maxVotes = count;
						bestSlotIndex = slotIndex;
					}
				}

				if (bestSlotIndex != -1)
				{
					bestLabel = m_slots[bestSlotIndex].label;
				}
			}

			// チャタリング対策.
			updateStability(bestLabel);

			return m_confirmedVowel;
		}

		void accumulateForLearning(const MFCC& mfcc)
		{
			m_learningBuffer.push_back(mfcc);
		}

		void resetLearningBuffer()
		{
			m_learningBuffer.clear();
		}

		bool commitLearning(int32 slotIndex)
		{
			if (slotIndex < 0 || slotIndex >= (int32) m_slots.size())
				return false;
			if (m_learningBuffer.size() <= 60)
				return false;

			m_slots[slotIndex].samples = m_learningBuffer;
			m_slots[slotIndex].isRecorded = true;
			return true;
		}

		int32 getLearningSampleCount() const { return static_cast<int32>(m_learningBuffer.size()); }
		Array<LearningSlot>& getSlots() { return m_slots; }
		const Array<LearningSlot>& getSlots() const { return m_slots; }

		bool isAllRecorded() const
		{
			return std::all_of(m_slots.begin(), m_slots.end(), [](const auto& s)
			{
				return s.isRecorded;
			});
		}

		String getPotentialVowel() const { return m_potentialVowel; }

	private:
		Array<LearningSlot> m_slots;
		Array<MFCC> m_learningBuffer;

		String m_potentialVowel = U"";
		String m_confirmedVowel = U"";
		int32 m_stabilityCount = 0;

		void updateStability(const String& currentBest)
		{
			if (currentBest != U"" && currentBest == m_potentialVowel)
				m_stabilityCount++;
			else
			{
				m_potentialVowel = currentBest;
				m_stabilityCount = 0;
			}

			if (m_stabilityCount > Config::StabilityFrames)
				m_confirmedVowel = m_potentialVowel;
		}
	};
} // namespace GameSystem

// ============================================================================
// 4. UserInterface: 画面UI制御
//    学習画面とゲーム画面の具体的な描画・入力フローを管理します.
// ============================================================================

namespace UserInterface
{
	using namespace AudioCore;
	using namespace GameSystem;
	using namespace DanmakuCore;

	class AppUI
	{
	public:
		AppUI()
			: m_player(Vec2{400, 500}), m_font(40), m_smallFont(20), m_enemyPos(400, 150)
		{
			m_bulletCurtain.start();
		}

		void update(const Microphone& mic, MFCCAnalyzer& analyzer)
		{
			const MFCC mfcc = analyzer.analyze(mic);
			const double rms = mic.rootMeanSquare();

			if (m_isGameMode)
				updateGamePhase(mfcc, rms);
			else
				updateLearningPhase(mfcc, rms);
		}

	private:
		bool m_isGameMode = false;
		int32 m_selectedSlotIndex = 0;
		bool m_isMousePressed = false;
		bool m_wasMousePressed = false;

		Player m_player;
		VoiceCommandSystem m_voiceSystem;
		Font m_font;
		Font m_smallFont;

		// 弾幕ゲーム関連.
		BulletCurtain m_bulletCurtain;
		Vec2 m_enemyPos;
		Texture m_enemyTexture{U"👾"_emoji};
		Effect m_effect;
		int32 m_score = 0;
		double m_enemyTime = 0.0;
		double m_enemyTargetX = 400.0;
		double m_enemyMoveStartX = 400.0;
		double m_enemyNextTargetX = 400.0;
		double m_lastEnemyCycleTime = -1.0;

		// 敵と弾幕を初期化する
		void resetEnemyAndBullets()
		{
			m_enemyTime = 0.0;
			m_enemyPos = Vec2(400.0, 150.0);
			m_enemyTargetX = 400.0;
			m_enemyMoveStartX = 400.0;
			m_enemyNextTargetX = 400.0;
			m_lastEnemyCycleTime = -1.0;
			m_bulletCurtain.clear();
			m_bulletCurtain.start();
		}

		void updateLearningPhase(const MFCC& mfcc, double rms)
		{
			Scene::SetBackground(Palette::Darkgray);
			m_font(U"学習モード: 音声を登録").drawAt(Scene::Width() / 2, 50, Palette::White);

			auto& slots = m_voiceSystem.getSlots();

			// スロット数に合わせて動的にレイアウト計算.
			const int32 slotCount = static_cast<int32>(slots.size());
			const int32 boxSize = 100;
			const int32 gap = 20;
			// 全体の幅を計算してセンタリング.
			const int32 startX = (Scene::Width() - (boxSize * slotCount + gap * (slotCount - 1))) / 2;
			const int32 startY = 200;

			// ナビゲーションボタン用のレイアウト
			const Rect prevBtn = Rect{Arg::center(startX - 80, startY + 50), 60, 60};
			const Rect nextBtn = Rect{Arg::center(Scene::Width() - startX + 80, startY + 50), 60, 60};

			// 前へボタン
			prevBtn.rounded(10).draw(prevBtn.mouseOver() ? Palette::Lightblue : Palette::Steelblue);
			m_smallFont(U"←").drawAt(prevBtn.center(), Palette::White);
			if (prevBtn.leftClicked())
				m_selectedSlotIndex = (m_selectedSlotIndex + (int32) slots.size() - 1) % slots.size();

			// 次へボタン
			nextBtn.rounded(10).draw(nextBtn.mouseOver() ? Palette::Lightblue : Palette::Steelblue);
			m_smallFont(U"→").drawAt(nextBtn.center(), Palette::White);
			if (nextBtn.leftClicked())
				m_selectedSlotIndex = (m_selectedSlotIndex + 1) % slots.size();

			// 音声判定を実行（学習用）
			String detectedCommand = m_voiceSystem.detectCommand(mfcc, rms);

			for (int32 i : step(slots.size()))
			{
				const Rect box{startX + i * (boxSize + gap), startY, boxSize, boxSize};
				const bool isSelected = (i == m_selectedSlotIndex);
				const bool isNoiseSlot = (slots[i].label == U"雑音");
				const bool isDetected = (detectedCommand == slots[i].label && !detectedCommand.isEmpty());

				if (box.leftClicked())
					m_selectedSlotIndex = i;

				// 判定されたボタンは異なる見た目でハイライト
				if (isDetected)
				{
					box.rounded(10).draw(ColorF{0.5, 0.8, 0.5});
					box.rounded(10).drawFrame(4, Palette::Limegreen);
				}
				else
				{
					box.rounded(10).draw(isSelected ? ColorF{0.3, 0.3, 0.4} : ColorF{0.2});
					if (isSelected)
						box.rounded(10).drawFrame(4, Palette::Skyblue);
				}

				m_font(slots[i].label).drawAt(box.center(), Palette::White);
				if (slots[i].isRecorded)
					m_smallFont(U"OK").drawAt(box.bottomCenter().movedBy(0, -20), Palette::Lightgreen);

				// 録音中でデータがバッファにあれば表示し続ける.
				if (isSelected && m_voiceSystem.getLearningSampleCount() > 0)
				{
					const double progress = Min(m_voiceSystem.getLearningSampleCount() / 60.0, 1.0);
					Circle{box.center(), 40}.drawArc(0_deg, 360_deg * progress, 4, 0, Palette::Orange);
				}
			}

			handleRecordingLogic(mfcc, rms);

			// 操作方法の表示
			m_smallFont(U"操作方法：選択したボックスをマウスで押しながら音声を話してください").drawAt(Scene::Width() / 2, 120, Palette::White);

			if (m_voiceSystem.isAllRecorded())
			{
				const Rect startBtn = Rect{Arg::center(Scene::Width() / 2, 450), 300, 60};
				startBtn.rounded(10).draw(startBtn.mouseOver() ? Palette::Orange : Palette::Darkorange);
				m_font(U"ゲーム開始").drawAt(startBtn.center(), Palette::White);

				if (startBtn.leftClicked())
				{
					m_isGameMode = true;
					m_player.reset(Vec2{400, 500});
					m_score = 0;
					resetEnemyAndBullets();
				}
			}
			else
			{
				m_smallFont(U"すべての音声を登録してください").drawAt(Scene::Width() / 2, 450, Palette::Gray);
			}
		}

		void handleRecordingLogic(const MFCC& mfcc, double rms)
		{
			auto& slots = m_voiceSystem.getSlots();
			const int32 slotCount = static_cast<int32>(slots.size());
			const int32 boxSize = 100;
			const int32 gap = 20;
			const int32 startX = (Scene::Width() - (boxSize * slotCount + gap * (slotCount - 1))) / 2;
			const int32 startY = 200;

			const Rect selectedBox{startX + m_selectedSlotIndex * (boxSize + gap), startY, boxSize, boxSize};
			bool isNoiseSlot = (slots[m_selectedSlotIndex].label == U"雑音");

			m_isMousePressed = MouseL.pressed();

			// マウスボタンが押された時点でバッファをリセット
			if (m_isMousePressed && !m_wasMousePressed)
			{
				m_voiceSystem.resetLearningBuffer();
			}

			// 選択ボックス内でマウスが押されている間、音声を蓄積
			if (selectedBox.mouseOver() && m_isMousePressed && (isNoiseSlot || rms > VolumeToRMS(Config::InputVolumeThreshold)) && !mfcc.isUnset())
			{
				m_voiceSystem.accumulateForLearning(mfcc);
			}

			// マウスボタンが離された時に学習を確定
			if (!m_isMousePressed && m_wasMousePressed)
			{
				bool success = m_voiceSystem.commitLearning(m_selectedSlotIndex);
				if (success)
				{
					// 学習成功時に未学習スロットの中で一番左のものに移動
					int32 leftmostUnrecorded = -1;
					for (int32 k = 0; k < (int32) slots.size(); ++k)
					{
						if (!slots[k].isRecorded)
						{
							leftmostUnrecorded = k;
							break;
						}
					}
					if (leftmostUnrecorded != -1)
					{
						m_selectedSlotIndex = leftmostUnrecorded;
					}
				}
				m_voiceSystem.resetLearningBuffer();
			}

			m_wasMousePressed = m_isMousePressed;
		}

		void updateGamePhase(const MFCC& mfcc, double rms)
		{
			Scene::SetBackground(ColorF{0.1, 0.2, 0.7});
			for (auto i : step(12))
			{
				const double a = Periodic::Sine0_1(2s, Scene::Time() - (2.0 / 12 * i));
				Rect{0, (i * 50), 800, 50}.draw(ColorF(1.0, a * 0.2));
			}

			String command = m_voiceSystem.detectCommand(mfcc, rms);
			String potential = m_voiceSystem.getPotentialVowel();

			// 敵の左右移動を更新（12秒ループ：11秒停止 + 1秒移動）
			m_enemyTime += Scene::DeltaTime();
			const double movementCyclePeriod = 12.0; // 12秒のループ
			const double stationaryDuration = 11.0; // 11秒停止
			const double movementDuration = 1.0; // 1秒移動

			const double timeInCycle = std::fmod(m_enemyTime, movementCyclePeriod);

			if (timeInCycle < stationaryDuration)
			{
				// 停止状態：敵は現在位置で待機
				// サイクル開始時に次の移動先をランダムに決定
				if (m_lastEnemyCycleTime >= stationaryDuration || m_lastEnemyCycleTime < 0)
				{
					// 前フレームで移動状態から停止状態に遷移した、または初回
					m_enemyNextTargetX = 280.0 + Random() * 240.0; // 280〜520の範囲
					m_enemyMoveStartX = m_enemyPos.x;
					m_enemyTargetX = m_enemyNextTargetX;
				}
				m_lastEnemyCycleTime = timeInCycle;
			}
			else
			{
				// 移動状態：開始位置から目標位置へ1秒かけて移動
				const double moveProgress = (timeInCycle - stationaryDuration) / movementDuration;
				m_enemyPos.x = m_enemyMoveStartX + (m_enemyTargetX - m_enemyMoveStartX) * moveProgress;
				m_lastEnemyCycleTime = timeInCycle;
			}

			m_bulletCurtain.update(m_enemyPos);
			m_player.update(command, Scene::DeltaTime());

			if (m_bulletCurtain.checkHit(m_player.getPos(), 8.0))
			{
				m_effect.add([pos = m_player.getPos()](double t)
				{
					const double t2 = (1.0 - t);
					Circle{pos, 10 + t * 70}.drawFrame(20 * t2, AlphaF(t2 * 0.5));
					return (t < 1.0);
				});

				m_player.reset(Vec2{400, 500});
				m_score = 0;
				resetEnemyAndBullets();
			}

			for (size_t i = 0; i < m_player.getBullets().size();)
			{
				const auto& bullet = m_player.getBullets()[i];
				if (bullet.pos.distanceFrom(m_enemyPos) < 100.0)
				{
					m_score += 100;
					m_effect.add([pos = bullet.pos](double t)
					{
						Circle{pos, t * 30}.drawFrame(2, Palette::Orange);
						return t < 0.5;
					});
					m_player.removeBullet(i);
				}
				else
				{
					++i;
				}
			}

			m_enemyTexture.resized(240).drawAt(m_enemyPos);
			m_bulletCurtain.draw();
			m_player.draw();
			m_effect.update();

			// プレイヤーの周りに「あいうえお」を表示
			const Vec2 playerPos = m_player.getPos();
			const double radius = 30.0;

			// あ: 中央, い: 左, う: 上, え: 右, お: 下
			Array<std::pair<String, Vec2>> vowelPositions = {
				{U"あ", Vec2(0, 0)},
				{U"い", Vec2(-1, 0)},
				{U"う", Vec2(0, -1)},
				{U"え", Vec2(1, 0)},
				{U"お", Vec2(0, 1)}
			};

			for (const auto& [vowel, direction] : vowelPositions)
			{
				const Vec2 pos = playerPos + radius * direction;
				const bool isActive = (command == vowel && !command.isEmpty());

				m_smallFont(vowel).drawAt(pos, isActive ? Palette::Yellow : Palette::White);
			}

			m_font(U"Score: {}"_fmt(m_score)).draw(20, 20, Palette::White);

			if (SimpleGUI::Button(U"再学習", Vec2{20, 80}))
			{
				m_isGameMode = false;
				m_score = 0;
			}
		}
	};
} // namespace UserInterface

// ============================================================================
// Main: エントリーポイント
// ============================================================================

void Main()
{
	Window::SetTitle(U"Voice Controller Danmaku");
	Window::Resize(800, 600);

	AudioCore::MFCCAnalyzer mfccAnalyzer{};
	Microphone mic{StartImmediately::Yes};
	UserInterface::AppUI appUI;

	while (System::Update())
	{
		if (System::EnumerateMicrophones().none([&](const auto& info)
		{
			return info.microphoneIndex == mic.microphoneIndex();
		}))
		{
			mic.open(StartImmediately::Yes);
		}
		appUI.update(mic, mfccAnalyzer);
	}
}
