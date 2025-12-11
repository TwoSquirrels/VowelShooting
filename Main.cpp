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
//    サンプルの弾幕システムを移植・調整したものです.
// ============================================================================

namespace DanmakuCore
{
	// 座標がシーン内にあるか判定する関数.
	bool isOutOfSceneArea(const Vec2& position)
	{
		constexpr int margin = 20; // すこし余裕を持たせる.
		return position.x < -margin || position.x > Scene::Width() + margin || position.y < -margin || position.y > Scene::Height() + margin;
	}

	// 敵弾.
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

	// 弾幕管理クラス.
	class BulletCurtain
	{
	public:
		BulletCurtain()
		{
			mWholePeriod = 10000;
			mBulletMap.emplace(eSpin, Array<EnemyBullet>());
			mBulletMap.emplace(eTail, Array<EnemyBullet>());
			mBulletMap.emplace(eSnow, Array<EnemyBullet>());
		}

		void clear()
		{
			mStopWatch.reset();
			for (auto& bullets : mBulletMap)
				bullets.second.clear();
		}

		void start() { mStopWatch.start(); }
		void pause() { mStopWatch.pause(); }

		void update(const Vec2& enemyPos)
		{
			if (mStopWatch.isPaused())
				return;

			updateEvents(enemyPos);
			updateBullets();
			eraseBullets();

			if (mStopWatch.ms() >= mWholePeriod)
			{
				mStopWatch.reset();
				mStopWatch.start();
			}
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
				constexpr int perNum = 5;
				for (int i = 0; i < perNum; ++i)
				{
					const double angle = 2 * Math::Pi / perNum * i;
					mBulletMap[eSpin].emplace_back(EnemyBullet(enemyPos, 120.f * Vec2(cos(angle), sin(angle)), Vec2::Zero(), 10));
				}
			}

			if (passedMs(700) && !passedMs(9000) && periodMs(150))
			{
				constexpr double speed = 1.35;
				for (const auto& spinBullet : mBulletMap[eSpin])
				{
					mBulletMap[eTail].emplace_back(EnemyBullet(spinBullet.pos, -speed * spinBullet.vel.rotated(-Math::Pi / 2), Vec2::Zero(), 7));
				}
			}

			if (periodMs(1000 + RandomInt32() % 150))
			{
				constexpr int perNum = 3;
				for (int i = 0; i < perNum; ++i)
				{
					const double genPos = Random() * 500.0 - 250.0;
					const double speed = Random() * 15.0 + 10.0;
					mBulletMap[eSnow].emplace_back(EnemyBullet(Vec2(genPos, -genPos), Vec2::Zero(), Vec2(speed, 1.2 * speed), 5));
				}
			}

			if (triggerMs(9000))
			{
				for (auto& spinBullet : mBulletMap[eSpin])
					spinBullet.acc = 3.0 * spinBullet.vel;
			}
		}

		void updateBullets()
		{
			for (auto& b : mBulletMap[eSpin])
			{
				b.vel.rotate(Math::Pi / 150);
				b.update();
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
		static constexpr double InputVolumeThreshold = 0.5;
		static constexpr double SimilarityThreshold = 0.85;
		static constexpr int32 StabilityFrames = 5;
		static constexpr double PlayerSpeed = 300.0; // 弾幕回避のため少し速めに.
		static constexpr double ShotSpeed = 800.0;
		static constexpr double ShotCoolTime = 0.15; // 秒.
	};

	// 自機ショット.
	struct PlayerBullet
	{
		Vec2 pos;
		Vec2 vel;
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
			// 移動処理 (い:左, う:上, え:右, お:下).
			if (command == U"い")
				m_pos.x -= Config::PlayerSpeed * deltaTime;
			else if (command == U"う")
				m_pos.y -= Config::PlayerSpeed * deltaTime;
			else if (command == U"え")
				m_pos.x += Config::PlayerSpeed * deltaTime;
			else if (command == U"お")
				m_pos.y += Config::PlayerSpeed * deltaTime;

			m_pos = m_pos.clamp(Scene::Rect());

			// ショット発射処理 (あ).
			m_shotTimer += deltaTime;
			if (command == U"あ" && m_shotTimer >= Config::ShotCoolTime)
			{
				m_bullets.emplace_back(PlayerBullet{m_pos, Vec2{0, -Config::ShotSpeed}});
				m_shotTimer = 0.0;
			}

			// ショットの移動と削除.
			for (auto& b : m_bullets)
				b.pos += b.vel * deltaTime;
			m_bullets.remove_if([](const PlayerBullet& b)
			{
				return b.pos.y < -50;
			});
		}

		void draw() const
		{
			// 自機描画.
			static const Texture playerTexture{U"🤖"_emoji};
			playerTexture.resized(50).flipped().drawAt(m_pos);

			// ショット描画.
			for (const auto& b : m_bullets)
			{
				Circle{b.pos, 8}.draw(Palette::Orange);
				Circle{b.pos, 5}.draw(Palette::Yellow);
			}
		}

		// リセット用.
		void reset(const Vec2& pos)
		{
			m_pos = pos;
			m_bullets.clear();
		}

		const Vec2& getPos() const { return m_pos; }
		const Array<PlayerBullet>& getBullets() const { return m_bullets; }

		// 弾が敵に当たったら消す処理.
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
		MFCC mfcc;
		bool isRecorded = false;
	};

	// 音声コマンドの管理システム.
	class VoiceCommandSystem
	{
	public:
		VoiceCommandSystem()
		{
			m_slots = {
				{U"あ", {}, false}, {U"い", {}, false}, {U"う", {}, false}, {U"え", {}, false}, {U"お", {}, false}};
		}

		String detectCommand(const MFCC& inputMFCC, double inputRMS)
		{
			if (inputRMS <= VolumeToRMS(Config::InputVolumeThreshold) || inputMFCC.isUnset())
			{
				m_potentialVowel = U"";
				m_stabilityCount = 0;
				m_confirmedVowel = U"";
				return m_confirmedVowel;
			}

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

			updateStability(bestLabel);
			return m_confirmedVowel;
		}

		void accumulateForLearning(const MFCC& mfcc)
		{
			if (m_learningBufferSum.isEmpty())
				m_learningBufferSum.resize(mfcc.feature.size(), 0.0);
			for (size_t i : step(m_learningBufferSum.size()))
				m_learningBufferSum[i] += mfcc.feature[i];
			m_learningSampleCount++;
		}

		void resetLearningBuffer()
		{
			m_learningBufferSum.clear();
			m_learningSampleCount = 0;
		}

		bool commitLearning(int32 slotIndex)
		{
			if (slotIndex < 0 || slotIndex >= (int32) m_slots.size())
				return false;
			if (m_learningSampleCount <= 10)
				return false;

			m_slots[slotIndex].mfcc = MFCC::Average(m_learningBufferSum, m_learningSampleCount);
			m_slots[slotIndex].isRecorded = true;
			return true;
		}

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

		String getPotentialVowel() const { return m_potentialVowel; }

	private:
		Array<LearningSlot> m_slots;
		Array<double> m_learningBufferSum;
		int32 m_learningSampleCount = 0;
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

		void updateLearningPhase(const MFCC& mfcc, double rms)
		{
			Scene::SetBackground(Palette::Darkgray);
			m_font(U"学習モード: 選択してSPACEキーを長押し").drawAt(Scene::Width() / 2, 50, Palette::White);

			auto& slots = m_voiceSystem.getSlots();

			if (KeyRight.down())
				m_selectedSlotIndex = (m_selectedSlotIndex + 1) % slots.size();
			if (KeyLeft.down())
				m_selectedSlotIndex = (m_selectedSlotIndex + (int32) slots.size() - 1) % slots.size();

			const int32 boxSize = 100;
			const int32 gap = 20;
			const int32 startX = (Scene::Width() - (boxSize * 5 + gap * 4)) / 2;
			const int32 startY = 200;

			for (int32 i : step(slots.size()))
			{
				const Rect box{startX + i * (boxSize + gap), startY, boxSize, boxSize};
				const bool isSelected = (i == m_selectedSlotIndex);
				const bool isRecordingInput = KeySpace.pressed() && rms > VolumeToRMS(Config::InputVolumeThreshold);
				const bool isRecordingNow = isSelected && isRecordingInput;

				if (box.leftClicked())
					m_selectedSlotIndex = i;

				box.rounded(10).draw(isSelected ? ColorF{0.3, 0.3, 0.4} : ColorF{0.2});
				if (isSelected)
					box.rounded(10).drawFrame(4, Palette::Skyblue);

				m_font(slots[i].label).drawAt(box.center(), Palette::White);
				if (slots[i].isRecorded)
					m_smallFont(U"OK").drawAt(box.bottomCenter().movedBy(0, -20), Palette::Lightgreen);

				if (isRecordingNow)
				{
					const double progress = Min(m_voiceSystem.getLearningSampleCount() / 60.0, 1.0);
					Circle{box.center(), 40}.drawArc(0_deg, 360_deg * progress, 4, 0, Palette::Orange);
				}
			}

			handleRecordingLogic(mfcc, rms);

			if (m_voiceSystem.isAllRecorded())
			{
				const Rect startBtn = Rect{Arg::center(Scene::Width() / 2, 450), 300, 60};
				startBtn.rounded(10).draw(startBtn.mouseOver() ? Palette::Orange : Palette::Darkorange);
				m_font(U"ゲーム開始").drawAt(startBtn.center(), Palette::White);

				if (startBtn.leftClicked() || KeyEnter.down())
				{
					m_isGameMode = true;
					// ゲーム開始時にリセット.
					m_bulletCurtain.clear();
					m_bulletCurtain.start();
					m_player.reset(Vec2{400, 500});
					m_score = 0;
				}
			}
			else
			{
				m_smallFont(U"すべての音声を登録してください").drawAt(Scene::Width() / 2, 450, Palette::Gray);
			}
		}

		void handleRecordingLogic(const MFCC& mfcc, double rms)
		{
			if (KeySpace.down())
				m_voiceSystem.resetLearningBuffer();
			if (KeySpace.pressed() && rms > VolumeToRMS(Config::InputVolumeThreshold) && !mfcc.isUnset())
			{
				m_voiceSystem.accumulateForLearning(mfcc);
			}
			if (KeySpace.up())
			{
				bool success = m_voiceSystem.commitLearning(m_selectedSlotIndex);
				if (success)
				{
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

		void updateGamePhase(const MFCC& mfcc, double rms)
		{
			// 背景.
			Scene::SetBackground(ColorF{0.1, 0.2, 0.7});
			for (auto i : step(12))
			{
				const double a = Periodic::Sine0_1(2s, Scene::Time() - (2.0 / 12 * i));
				Rect{0, (i * 50), 800, 50}.draw(ColorF(1.0, a * 0.2));
			}

			// コマンド判定.
			String command = m_voiceSystem.detectCommand(mfcc, rms);
			String potential = m_voiceSystem.getPotentialVowel();

			// 弾幕更新.
			m_bulletCurtain.update(m_enemyPos);

			// プレイヤー更新.
			m_player.update(command, Scene::DeltaTime());

			// 当たり判定: 敵弾 vs 自機.
			if (m_bulletCurtain.checkHit(m_player.getPos(), 4.0))
			{
				// 爆発エフェクト.
				m_effect.add([pos = m_player.getPos()](double t)
				{
					const double t2 = (1.0 - t);
					Circle{pos, 10 + t * 70}.drawFrame(20 * t2, AlphaF(t2 * 0.5));
					return (t < 1.0);
				});

				// ゲームオーバーリセット.
				m_player.reset(Vec2{400, 500});
				m_bulletCurtain.clear();
				m_bulletCurtain.start();
				m_score = 0;
			}

			// 当たり判定: 自機弾 vs 敵.
			// 簡易的な処理として、敵の中心付近に当たったらヒットとする.
			for (const auto& bullet : m_player.getBullets())
			{
				if (bullet.pos.distanceFrom(m_enemyPos) < 40.0) // 敵の当たり判定サイズ.
				{
					m_score += 100;
					m_effect.add([pos = bullet.pos](double t)
					{
						Circle{pos, t * 30}.drawFrame(2, Palette::Orange);
						return t < 0.5;
					});
					// 弾を消す処理は複雑になるので今回は省略(貫通)するか、または描画側で制御してもよい.
				}
			}

			// 描画.
			m_enemyTexture.resized(60).drawAt(m_enemyPos); // 敵.
			m_bulletCurtain.draw(); // 弾幕.
			m_player.draw(); // 自機.
			m_effect.update(); // エフェクト.

			// UI表示.
			if (command)
				m_font(command).drawAt(Scene::Center().movedBy(0, -200), Palette::White);
			else if (potential)
				m_font(potential).drawAt(Scene::Center().movedBy(0, -200), ColorF{1.0, 0.5});

			m_font(U"Score: {}"_fmt(m_score)).draw(20, 20, Palette::White);
			m_smallFont(U"あ:ショット  い:←  う:↑  え:→  お:↓").drawAt(Scene::Width() / 2, Scene::Height() - 30, Palette::White);

			// 再学習ボタン.
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
