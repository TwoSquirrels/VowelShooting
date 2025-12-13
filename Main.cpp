#include <Siv3D.hpp> // Siv3D v0.6.16

namespace AudioCore
{
	[[nodiscard]] double VolumeToRMS(const double volume)
	{
		return Clamp(Math::Pow(10.0, (volume - 1.0) * 5.0), 0.0, 1.0);
	}

	struct MFCC
	{
		Array<double> feature;

		[[nodiscard]] bool isUnset() const
		{
			return std::ranges::all_of(feature, [](const double x) { return x == 0.0; });
		}

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
		[[nodiscard]] static double freqToMel(const double freq)
		{
			return 1127.01 * Math::Log(1.0 + freq / 700.0);
		}

		[[nodiscard]] static double melToFreq(const double mel)
		{
			return 700.0 * (Math::Exp(mel / 1127.01) - 1.0);
		}

		explicit MFCCAnalyzer(
			const FFTSampleLength frames = FFTSampleLength::SL2K,
			const size_t melChannels = 40,
			const size_t mfccOrder = 12)
			: m_frames(frames)
			, m_melChannels(melChannels)
			, m_mfccOrder(mfccOrder)
			, m_f(256uLL << FromEnum(frames), 0.0f)
			, m_bin(melChannels + 2)
			, m_melSpectrum(melChannels)
			, m_melEnvelope(melChannels)
		{
		}

		[[nodiscard]] MFCC analyze(const Microphone& mic)
		{
			if (not mic.isLoop())
				throw Error{U"Microphone must be in loop mode."};

			if (not mic.isRecording() || mic.getBufferLength() < m_f.size())
				return MFCC{Array<double>(m_mfccOrder, 0.0)};

			const auto sampleRate = mic.getSampleRate();
			const auto& buffer = mic.getBuffer();
			const size_t writePos = mic.posSample();

			copyBufferWithPreEmphasis(buffer, writePos, mic.getBufferLength());
			applyHammingWindow();

			FFT::Analyze(m_fftResult, m_f.data(), m_f.size(), sampleRate, m_frames);

			computeMelFilterBank(sampleRate);
			return computeMFCC();
		}

	private:
		static constexpr float PreEmphasisCoeff = 0.96875f;

		FFTSampleLength m_frames;
		size_t m_melChannels;
		size_t m_mfccOrder;
		Array<float> m_f;
		Array<size_t> m_bin;
		Array<double> m_melSpectrum;
		Array<Vec2> m_melEnvelope;
		FFTResult m_fftResult;

		void copyBufferWithPreEmphasis(const Array<WaveSample>& buffer, size_t writePos, size_t bufferLength)
		{
			for (size_t pos : step(m_f.size()))
			{
				const size_t idx = (pos + writePos < m_f.size() ? bufferLength : 0) + pos + writePos - m_f.size();
				m_f[pos] = buffer[idx].left;
			}

			for (size_t i = m_f.size() - 1; i >= 1; --i)
				m_f[i] -= m_f[i - 1] * PreEmphasisCoeff;
		}

		void applyHammingWindow()
		{
			const size_t n = m_f.size();
			m_f.front() = 0.0f;
			m_f.back() = 0.0f;

			for (size_t i = 1; i < n - 1; ++i)
				m_f[i] *= static_cast<float>(0.54 - 0.46 * std::cos(2.0 * Math::Pi * i / (n - 1)));
		}

		void computeMelFilterBank(uint32 sampleRate)
		{
			const double melMax = freqToMel(static_cast<double>(sampleRate) / 2.0);
			const double melMin = freqToMel(0);
			const double deltaMel = (melMax - melMin) / static_cast<double>(m_melChannels + 1);

			for (size_t i : step(m_bin.size()))
				m_bin[i] = static_cast<size_t>((m_f.size() + 1) * melToFreq(melMin + i * deltaMel) / sampleRate);

			for (size_t i : step(m_melChannels))
			{
				m_melSpectrum[i] = 0.0;

				for (size_t j = m_bin[i]; j < m_bin[i + 1]; ++j)
					m_melSpectrum[i] += static_cast<double>(m_fftResult.buffer[j]) * (j - m_bin[i]) / (m_bin[i + 1] - m_bin[i]);

				for (size_t j = m_bin[i + 1]; j < m_bin[i + 2]; ++j)
					m_melSpectrum[i] += static_cast<double>(m_fftResult.buffer[j]) * (m_bin[i + 2] - j) / (m_bin[i + 2] - m_bin[i + 1]);

				m_melEnvelope[i] = {2.0 * m_bin[i + 1] / m_f.size(), m_melSpectrum[i] / (m_bin[i + 2] - m_bin[i])};
			}
		}

		[[nodiscard]] MFCC computeMFCC() const
		{
			MFCC mfcc{Array<double>(m_mfccOrder, 0.0)};

			for (size_t i = 1; i <= m_mfccOrder; ++i)
			{
				for (size_t j : step(m_melChannels))
				{
					mfcc.feature[i - 1] += Math::Log10(Math::Abs(m_melSpectrum[j]))
						* Math::Cos(Math::Pi * i * (j + 0.5) / m_melChannels) * 10;
				}
			}
			return mfcc;
		}
	};
}

namespace DanmakuCore
{
	namespace
	{
		constexpr int32 SceneMargin = 20;

		[[nodiscard]] bool isOutOfSceneArea(const Vec2& position)
		{
			return position.x < -SceneMargin
				|| position.x > Scene::Width() + SceneMargin
				|| position.y < -SceneMargin
				|| position.y > Scene::Height() + SceneMargin;
		}
	}

	struct EnemyBullet
	{
		Vec2 pos;
		Vec2 vel;
		Vec2 acc;
		double size;
		Stopwatch stopwatch{StartImmediately::Yes};

		EnemyBullet(const Vec2& pos_, const Vec2& vel_, const Vec2& acc_, double size_)
			: pos(pos_), vel(vel_), acc(acc_), size(size_)
		{
		}

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
			for (int key : {eSpin, eTail, eSnow})
				m_bulletMap.emplace(key, Array<EnemyBullet>());
		}

		void clear()
		{
			m_stopwatch.reset();
			for (auto& [key, bullets] : m_bulletMap)
				bullets.clear();
			m_prevEnemyPos = Vec2::Zero();
		}

		void start() { m_stopwatch.start(); }
		void pause() { m_stopwatch.pause(); }

		void update(const Vec2& enemyPos)
		{
			if (m_stopwatch.isPaused())
				return;

			updateEvents(enemyPos);
			updateBullets(enemyPos);
			eraseBullets();

			if (m_stopwatch.ms() >= WholePeriodMs)
			{
				m_stopwatch.reset();
				m_stopwatch.start();
			}

			m_prevEnemyPos = enemyPos;
		}

		void draw() const
		{
			for (const auto& b : m_bulletMap.at(eSnow))
				Circle{b.pos, b.size}.draw(Palette::White);
			for (const auto& b : m_bulletMap.at(eTail))
				Circle{b.pos, b.size}.draw(Palette::Hotpink);
			for (const auto& b : m_bulletMap.at(eSpin))
				Circle{b.pos, b.size}.draw(Palette::White);
		}

		[[nodiscard]] bool checkHit(const Vec2& pos, double size) const
		{
			for (const auto& [key, bullets] : m_bulletMap)
			{
				for (const auto& bullet : bullets)
				{
					if (bullet.pos.distanceFrom(pos) <= bullet.size + size)
						return true;
				}
			}
			return false;
		}

	private:
		enum BulletKey { eSpin, eTail, eSnow };

		static constexpr int32 WholePeriodMs = 10000;

		HashTable<int, Array<EnemyBullet>> m_bulletMap;
		Stopwatch m_stopwatch;
		Vec2 m_prevEnemyPos = Vec2::Zero();

		[[nodiscard]] bool triggerMs(int32 timePoint) const
		{
			return std::abs(m_stopwatch.ms() - timePoint) <= Scene::DeltaTime() * 1000.0 / 1.7;
		}

		[[nodiscard]] bool periodMs(int32 period) const
		{
			const int32 now = m_stopwatch.ms();
			const double deltams = Scene::DeltaTime() * 1000.0 / 1.5;
			return std::abs(now % period - period) <= deltams || now % period <= deltams;
		}

		[[nodiscard]] bool passedMs(int32 timePoint) const
		{
			return m_stopwatch.ms() >= timePoint;
		}

		void updateEvents(const Vec2& enemyPos)
		{
			if (triggerMs(500))
			{
				constexpr int perNum = 3;
				for (int i = 0; i < perNum; ++i)
				{
					const double angle = 2 * Math::Pi / perNum * i;
					m_bulletMap[eSpin].emplace_back(enemyPos, 80.0 * Vec2{std::cos(angle), std::sin(angle)}, Vec2::Zero(), 10.0);
				}
			}

			if (passedMs(700) && !passedMs(9000) && periodMs(350))
			{
				for (const auto& spinBullet : m_bulletMap[eSpin])
					m_bulletMap[eTail].emplace_back(spinBullet.pos, -spinBullet.vel.rotated(-Math::HalfPi), Vec2::Zero(), 7.0);
			}

			if (periodMs(2000 + Random(300)))
			{
				constexpr int perNum = 2;
				for (int i = 0; i < perNum; ++i)
				{
					const double genPos = Random(-250.0, 250.0);
					const double speed = Random(6.0, 16.0);
					m_bulletMap[eSnow].emplace_back(Vec2{genPos, -genPos}, Vec2::Zero(), Vec2{speed, 1.2 * speed}, 5.0);
				}
			}

			if (triggerMs(9000))
			{
				for (auto& spinBullet : m_bulletMap[eSpin])
					spinBullet.acc = 2.0 * spinBullet.vel;
			}
		}

		void updateBullets(const Vec2& enemyPos)
		{
			const Vec2 enemyVelocity = enemyPos - m_prevEnemyPos;

			for (auto& b : m_bulletMap[eSpin])
			{
				b.vel.rotate(Math::Pi / 150);
				b.update();
				b.pos += enemyVelocity;
			}

			for (auto& b : m_bulletMap[eTail])
			{
				const double rotationSpeed = (b.stopwatch.ms() < 3000) ? Math::Pi / 270 : Math::Pi / 450;
				b.vel.rotate(rotationSpeed);
				b.update();
			}

			for (auto& b : m_bulletMap[eSnow])
			{
				b.vel.rotate(-Math::Pi / 2200);
				b.update();
			}
		}

		void eraseBullets()
		{
			m_bulletMap[eSpin].remove_if([](const EnemyBullet& b) { return isOutOfSceneArea(b.pos); });
			m_bulletMap[eTail].remove_if([](const EnemyBullet& b) { return isOutOfSceneArea(b.pos); });
			m_bulletMap[eSnow].remove_if([](const EnemyBullet& b)
			{
				return b.pos.x > Scene::Width() + SceneMargin || b.pos.y > Scene::Height() + SceneMargin;
			});
		}
	};
}

namespace GameSystem
{
	enum class GamePhase { Learning, Playing, GameOver };

	struct Config
	{
		static constexpr double InputVolumeThreshold = 0.1;
		static constexpr int32 K_Nearest = 7;
		static constexpr int32 StabilityFrames = 5;
		static constexpr int32 MinLearningSamples = 60;

		static constexpr double PlayerSpeed = 250.0;
		static constexpr double PlayerHitboxSize = 8.0;
		static constexpr double PlayerDisplaySize = 15.0;
		static constexpr double ShotSpeed = 500.0;
		static constexpr double ShotCoolTime = 0.15;
		static constexpr double VowelDisplayRadius = 30.0;
		static constexpr Vec2 PlayerInitialPos{400.0, 500.0};

		static constexpr Vec2 EnemyInitialPos{400.0, 150.0};
		static constexpr double EnemyHitboxSize = 100.0;
		static constexpr double EnemyDisplaySize = 240.0;
		static constexpr double EnemyMovementCyclePeriod = 12.0;
		static constexpr double EnemyStationaryDuration = 11.0;
		static constexpr double EnemyMovementDuration = 1.0;
		static constexpr double EnemyMoveRangeMin = 280.0;
		static constexpr double EnemyMoveRangeMax = 520.0;

		static constexpr int32 SlotBoxSize = 100;
		static constexpr int32 SlotGap = 20;
		static constexpr int32 ScorePerHit = 100;
		static constexpr Size WindowSize{800, 600};
	};

	struct PlayerBullet
	{
		Vec2 pos;
		Vec2 vel;
	};

	class Player
	{
	public:
		Player(const Vec2& initialPos, const Font& smallFont)
			: m_pos(initialPos)
			, m_smallFont(smallFont)
		{
		}

		void update(const String& command, double deltaTime)
		{
			updateMovement(command, deltaTime);
			updateShooting(command, deltaTime);
			updateBullets(deltaTime);
		}

		void draw() const
		{
			Circle{m_pos, Config::PlayerDisplaySize}.draw(ColorF{0.25, 0.25, 0.28});

			for (const auto& b : m_bullets)
			{
				Circle{b.pos, 8}.draw(Palette::Orange);
				Circle{b.pos, 5}.draw(Palette::Yellow);
			}
		}

		void drawVowelIndicators(const String& activeCommand) const
		{
			static const Array<std::pair<String, Vec2>> VowelDirections = {
				{U"あ", Vec2::Zero()},
				{U"い", Vec2::Left()},
				{U"う", Vec2::Up()},
				{U"え", Vec2::Right()},
				{U"お", Vec2::Down()}
			};

			for (const auto& [vowel, direction] : VowelDirections)
			{
				const Vec2 pos = m_pos + Config::VowelDisplayRadius * direction;
				const bool isActive = (activeCommand == vowel);
				m_smallFont(vowel).drawAt(pos, isActive ? Palette::Yellow : Palette::White);
			}
		}

		void reset(const Vec2& pos)
		{
			m_pos = pos;
			m_bullets.clear();
			m_shotTimer = 0.0;
		}

		[[nodiscard]] const Vec2& getPos() const { return m_pos; }
		[[nodiscard]] const Array<PlayerBullet>& getBullets() const { return m_bullets; }

		void removeBulletAt(size_t index)
		{
			if (index < m_bullets.size())
				m_bullets.remove_at(index);
		}

	private:
		Vec2 m_pos;
		Array<PlayerBullet> m_bullets;
		double m_shotTimer = 0.0;
		const Font& m_smallFont;

		void updateMovement(const String& command, double deltaTime)
		{
			const double delta = Config::PlayerSpeed * deltaTime;

			if (command == U"い")      m_pos.x -= delta;
			else if (command == U"う") m_pos.y -= delta;
			else if (command == U"え") m_pos.x += delta;
			else if (command == U"お") m_pos.y += delta;

			m_pos = m_pos.clamp(Scene::Rect());
		}

		void updateShooting(const String& command, double deltaTime)
		{
			m_shotTimer += deltaTime;

			if (command == U"あ" && m_shotTimer >= Config::ShotCoolTime)
			{
				m_bullets.emplace_back(PlayerBullet{m_pos, Vec2{0, -Config::ShotSpeed}});
				m_shotTimer = 0.0;
			}
		}

		void updateBullets(double deltaTime)
		{
			for (auto& b : m_bullets)
				b.pos += b.vel * deltaTime;

			m_bullets.remove_if([](const PlayerBullet& b) { return b.pos.y < -50; });
		}
	};

	class Enemy
	{
	public:
		Enemy()
			: m_pos(Config::EnemyInitialPos)
			, m_texture(U"👾"_emoji)
		{
		}

		void update(double deltaTime)
		{
			m_time += deltaTime;
			updateMovement();
		}

		void draw() const
		{
			m_texture.resized(Config::EnemyDisplaySize).drawAt(m_pos);
		}

		void reset()
		{
			m_time = 0.0;
			m_pos = Config::EnemyInitialPos;
			m_targetX = Config::EnemyInitialPos.x;
			m_moveStartX = Config::EnemyInitialPos.x;
			m_lastCycleTime = -1.0;
		}

		[[nodiscard]] const Vec2& getPos() const { return m_pos; }

		[[nodiscard]] bool checkHitWithBullet(const Vec2& bulletPos) const
		{
			return bulletPos.distanceFrom(m_pos) < Config::EnemyHitboxSize;
		}

	private:
		Vec2 m_pos;
		Texture m_texture;
		double m_time = 0.0;
		double m_targetX = Config::EnemyInitialPos.x;
		double m_moveStartX = Config::EnemyInitialPos.x;
		double m_lastCycleTime = -1.0;

		void updateMovement()
		{
			const double timeInCycle = std::fmod(m_time, Config::EnemyMovementCyclePeriod);

			if (timeInCycle < Config::EnemyStationaryDuration)
			{
				if (m_lastCycleTime >= Config::EnemyStationaryDuration || m_lastCycleTime < 0)
				{
					m_targetX = Random(Config::EnemyMoveRangeMin, Config::EnemyMoveRangeMax);
					m_moveStartX = m_pos.x;
				}
			}
			else
			{
				const double progress = (timeInCycle - Config::EnemyStationaryDuration) / Config::EnemyMovementDuration;
				m_pos.x = Math::Lerp(m_moveStartX, m_targetX, progress);
			}

			m_lastCycleTime = timeInCycle;
		}
	};

	struct LearningSlot
	{
		String label;
		Array<AudioCore::MFCC> samples;
		bool isRecorded = false;
	};

	/// @brief k-NN による音声コマンド認識システム
	class VoiceCommandSystem
	{
	public:
		VoiceCommandSystem()
			: m_slots({
				{U"雑音", {}, false},
				{U"あ", {}, false},
				{U"い", {}, false},
				{U"う", {}, false},
				{U"え", {}, false},
				{U"お", {}, false}
			})
		{
		}

		[[nodiscard]] String detectCommand(const AudioCore::MFCC& inputMFCC)
		{
			if (inputMFCC.isUnset())
			{
				resetStability();
				return m_confirmedVowel;
			}

			updateStability(findNearestLabel(inputMFCC));
			return m_confirmedVowel;
		}

		void accumulateForLearning(const AudioCore::MFCC& mfcc)
		{
			m_learningBuffer.push_back(mfcc);
		}

		void resetLearningBuffer() { m_learningBuffer.clear(); }

		bool commitLearning(int32 slotIndex)
		{
			if (slotIndex < 0 || slotIndex >= static_cast<int32>(m_slots.size()))
				return false;

			if (static_cast<int32>(m_learningBuffer.size()) <= Config::MinLearningSamples)
				return false;

			m_slots[slotIndex].samples = m_learningBuffer;
			m_slots[slotIndex].isRecorded = true;
			return true;
		}

		[[nodiscard]] int32 getLearningSampleCount() const
		{
			return static_cast<int32>(m_learningBuffer.size());
		}

		[[nodiscard]] Array<LearningSlot>& getSlots() { return m_slots; }
		[[nodiscard]] const Array<LearningSlot>& getSlots() const { return m_slots; }

		[[nodiscard]] bool isAllRecorded() const
		{
			return std::ranges::all_of(m_slots, [](const auto& s) { return s.isRecorded; });
		}

		void resetDetectionState()
		{
			m_potentialVowel = U"";
			m_confirmedVowel = U"";
			m_stabilityCount = 0;
		}

		[[nodiscard]] int32 findFirstUnrecordedSlotIndex() const
		{
			for (int32 i = 0; i < static_cast<int32>(m_slots.size()); ++i)
			{
				if (!m_slots[i].isRecorded)
					return i;
			}
			return -1;
		}

	private:
		Array<LearningSlot> m_slots;
		Array<AudioCore::MFCC> m_learningBuffer;
		String m_potentialVowel;
		String m_confirmedVowel;
		int32 m_stabilityCount = 0;

		[[nodiscard]] String findNearestLabel(const AudioCore::MFCC& inputMFCC) const
		{
			struct Neighbor
			{
				double distSq;
				int32 slotIndex;
			};

			Array<Neighbor> neighbors;
			neighbors.reserve(m_slots.size() * Config::MinLearningSamples);

			for (int32 i = 0; i < static_cast<int32>(m_slots.size()); ++i)
			{
				for (const auto& sample : m_slots[i].samples)
					neighbors.push_back({inputMFCC.distSq(sample), i});
			}

			if (neighbors.isEmpty())
				return U"";

			const size_t k = Min<size_t>(Config::K_Nearest, neighbors.size());
			std::partial_sort(neighbors.begin(), neighbors.begin() + k, neighbors.end(),
				[](const Neighbor& a, const Neighbor& b) { return a.distSq < b.distSq; });

			HashTable<int32, int32> votes;
			for (size_t i = 0; i < k; ++i)
				votes[neighbors[i].slotIndex]++;

			int32 bestSlotIndex = -1;
			int32 maxVotes = 0;

			for (const auto& [slotIndex, count] : votes)
			{
				if (count > maxVotes)
				{
					maxVotes = count;
					bestSlotIndex = slotIndex;
				}
			}

			return (bestSlotIndex >= 0) ? m_slots[bestSlotIndex].label : U"";
		}

		void resetStability()
		{
			m_potentialVowel = U"";
			m_confirmedVowel = U"";
			m_stabilityCount = 0;
		}

		/// @brief チャタリング防止のため、同じ結果が連続したときのみ確定する
		void updateStability(const String& currentBest)
		{
			if (!currentBest.isEmpty() && currentBest == m_potentialVowel)
			{
				++m_stabilityCount;
			}
			else
			{
				m_potentialVowel = currentBest;
				m_stabilityCount = 0;
			}

			if (m_stabilityCount > Config::StabilityFrames)
				m_confirmedVowel = m_potentialVowel;
		}
	};
}

namespace UserInterface
{
	using namespace GameSystem;
	using namespace DanmakuCore;

	class AppUI
	{
	public:
		AppUI()
			: m_font(40)
			, m_smallFont(20)
			, m_player(Config::PlayerInitialPos, m_smallFont)
		{
			m_bulletCurtain.start();
		}

		void update(const Microphone& mic, AudioCore::MFCCAnalyzer& analyzer)
		{
			const auto mfcc = analyzer.analyze(mic);
			const double rms = mic.rootMeanSquare();

			switch (m_phase)
			{
			case GamePhase::Learning:
				updateLearningPhase(mfcc, rms);
				break;
			case GamePhase::Playing:
				updatePlayingPhase(mfcc);
				break;
			case GamePhase::GameOver:
				updateGameOverPhase();
				break;
			}
		}

	private:
		Font m_font;
		Font m_smallFont;

		GamePhase m_phase = GamePhase::Learning;
		int32 m_selectedSlotIndex = 0;
		bool m_wasMousePressed = false;
		int32 m_score = 0;

		Player m_player;
		Enemy m_enemy;
		VoiceCommandSystem m_voiceSystem;
		BulletCurtain m_bulletCurtain;
		Effect m_effect;

		void resetGame()
		{
			m_player.reset(Config::PlayerInitialPos);
			m_enemy.reset();
			m_bulletCurtain.clear();
			m_bulletCurtain.start();
			m_score = 0;
			m_voiceSystem.resetDetectionState();
		}

		void transitionTo(GamePhase newPhase)
		{
			m_phase = newPhase;
			if (newPhase == GamePhase::Playing)
				resetGame();
		}

		void drawGameBackground() const
		{
			Scene::SetBackground(ColorF{0.1, 0.2, 0.7});

			for (int32 i = 0; i < 12; ++i)
			{
				const double alpha = Periodic::Sine0_1(2s, Scene::Time() - (2.0 / 12 * i));
				Rect{0, i * 50, Config::WindowSize.x, 50}.draw(ColorF{1.0, alpha * 0.2});
			}
		}

		void updateLearningPhase(const AudioCore::MFCC& mfcc, double rms)
		{
			Scene::SetBackground(Palette::Darkgray);

			drawLearningHeader();
			drawLearningSlots(mfcc);
			handleRecordingLogic(mfcc, rms);
			drawLearningFooter();
		}

		void drawLearningHeader() const
		{
			m_font(U"学習モード: 音声を登録").drawAt(Scene::Width() / 2, 50, Palette::White);
			m_smallFont(U"操作方法：選択したボックスをマウスで押しながら音声を話してください")
				.drawAt(Scene::Width() / 2, 120, Palette::White);
		}

		[[nodiscard]] std::pair<int32, int32> calcSlotLayoutOrigin(size_t slotCount) const
		{
			const int32 totalWidth = Config::SlotBoxSize * static_cast<int32>(slotCount)
				+ Config::SlotGap * (static_cast<int32>(slotCount) - 1);
			return {(Scene::Width() - totalWidth) / 2, 200};
		}

		[[nodiscard]] Rect calcSlotBox(int32 index, int32 startX, int32 startY) const
		{
			return Rect{
				startX + index * (Config::SlotBoxSize + Config::SlotGap),
				startY,
				Config::SlotBoxSize,
				Config::SlotBoxSize
			};
		}

		void drawLearningSlots(const AudioCore::MFCC& mfcc)
		{
			const auto& slots = m_voiceSystem.getSlots();
			const auto [startX, startY] = calcSlotLayoutOrigin(slots.size());

			drawNavigationButtons(startX, startY, slots.size());

			const String detectedCommand = m_voiceSystem.detectCommand(mfcc);

			for (int32 i = 0; i < static_cast<int32>(slots.size()); ++i)
			{
				const Rect box = calcSlotBox(i, startX, startY);
				const bool isSelected = (i == m_selectedSlotIndex);
				const bool isDetected = (detectedCommand == slots[i].label);

				if (box.leftClicked())
					m_selectedSlotIndex = i;

				drawSlotBox(box, slots[i], isSelected, isDetected);
				drawSlotProgress(box, isSelected);
			}
		}

		void drawNavigationButtons(int32 startX, int32 startY, size_t slotCount)
		{
			const Rect prevBtn{Arg::center(startX - 80, startY + 50), 60, 60};
			const Rect nextBtn{Arg::center(Scene::Width() - startX + 80, startY + 50), 60, 60};

			prevBtn.rounded(10).draw(prevBtn.mouseOver() ? Palette::Lightblue : Palette::Steelblue);
			m_smallFont(U"←").drawAt(prevBtn.center(), Palette::White);
			if (prevBtn.leftClicked())
				m_selectedSlotIndex = (m_selectedSlotIndex + static_cast<int32>(slotCount) - 1) % slotCount;

			nextBtn.rounded(10).draw(nextBtn.mouseOver() ? Palette::Lightblue : Palette::Steelblue);
			m_smallFont(U"→").drawAt(nextBtn.center(), Palette::White);
			if (nextBtn.leftClicked())
				m_selectedSlotIndex = (m_selectedSlotIndex + 1) % slotCount;
		}

		void drawSlotBox(const Rect& box, const LearningSlot& slot, bool isSelected, bool isDetected) const
		{
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

			m_font(slot.label).drawAt(box.center(), Palette::White);

			if (slot.isRecorded)
				m_smallFont(U"OK").drawAt(box.bottomCenter().movedBy(0, -20), Palette::Lightgreen);
		}

		void drawSlotProgress(const Rect& box, bool isSelected) const
		{
			if (isSelected && m_voiceSystem.getLearningSampleCount() > 0)
			{
				const double progress = Min(
					m_voiceSystem.getLearningSampleCount() / static_cast<double>(Config::MinLearningSamples),
					1.0
				);
				Circle{box.center(), 40}.drawArc(0_deg, 360_deg * progress, 4, 0, Palette::Orange);
			}
		}

		void handleRecordingLogic(const AudioCore::MFCC& mfcc, double rms)
		{
			const auto& slots = m_voiceSystem.getSlots();
			const auto [startX, startY] = calcSlotLayoutOrigin(slots.size());
			const Rect selectedBox = calcSlotBox(m_selectedSlotIndex, startX, startY);
			const bool isNoiseSlot = (slots[m_selectedSlotIndex].label == U"雑音");
			const bool isMousePressed = MouseL.pressed();

			if (isMousePressed && !m_wasMousePressed)
				m_voiceSystem.resetLearningBuffer();

			const bool shouldAccumulate = selectedBox.mouseOver()
				&& isMousePressed
				&& (isNoiseSlot || rms > AudioCore::VolumeToRMS(Config::InputVolumeThreshold))
				&& !mfcc.isUnset();

			if (shouldAccumulate)
				m_voiceSystem.accumulateForLearning(mfcc);

			if (!isMousePressed && m_wasMousePressed)
			{
				if (m_voiceSystem.commitLearning(m_selectedSlotIndex))
				{
					const int32 nextSlot = m_voiceSystem.findFirstUnrecordedSlotIndex();
					if (nextSlot >= 0)
						m_selectedSlotIndex = nextSlot;
				}
				m_voiceSystem.resetLearningBuffer();
			}

			m_wasMousePressed = isMousePressed;
		}

		void drawLearningFooter()
		{
			if (m_voiceSystem.isAllRecorded())
			{
				const Rect startBtn{Arg::center(Scene::Width() / 2, 450), 300, 60};
				startBtn.rounded(10).draw(startBtn.mouseOver() ? Palette::Orange : Palette::Darkorange);
				m_font(U"ゲーム開始").drawAt(startBtn.center(), Palette::White);

				if (startBtn.leftClicked())
					transitionTo(GamePhase::Playing);
			}
			else
			{
				m_smallFont(U"すべての音声を登録してください").drawAt(Scene::Width() / 2, 450, Palette::Gray);
			}
		}

		void updatePlayingPhase(const AudioCore::MFCC& mfcc)
		{
			drawGameBackground();

			const String command = m_voiceSystem.detectCommand(mfcc);

			m_enemy.update(Scene::DeltaTime());
			m_bulletCurtain.update(m_enemy.getPos());
			m_player.update(command, Scene::DeltaTime());

			checkCollisions();

			m_enemy.draw();
			m_bulletCurtain.draw();
			m_player.draw();
			m_player.drawVowelIndicators(command);
			m_effect.update();

			drawPlayingUI();
		}

		void checkCollisions()
		{
			if (m_bulletCurtain.checkHit(m_player.getPos(), Config::PlayerHitboxSize))
			{
				addPlayerDeathEffect();
				transitionTo(GamePhase::GameOver);
				return;
			}

			const auto& bullets = m_player.getBullets();
			for (size_t i = 0; i < bullets.size();)
			{
				if (m_enemy.checkHitWithBullet(bullets[i].pos))
				{
					m_score += Config::ScorePerHit;
					addBulletHitEffect(bullets[i].pos);
					m_player.removeBulletAt(i);
				}
				else
				{
					++i;
				}
			}
		}

		void addPlayerDeathEffect()
		{
			m_effect.add([pos = m_player.getPos()](double t)
			{
				const double fade = 1.0 - t;
				Circle{pos, 10 + t * 70}.drawFrame(20 * fade, AlphaF(fade * 0.5));
				return t < 1.0;
			});
		}

		void addBulletHitEffect(const Vec2& pos)
		{
			m_effect.add([pos](double t)
			{
				Circle{pos, t * 30}.drawFrame(2, Palette::Orange);
				return t < 0.5;
			});
		}

		void drawPlayingUI()
		{
			m_font(U"Score: {}"_fmt(m_score)).draw(20, 20, Palette::White);

			if (SimpleGUI::Button(U"再学習", Vec2{20, 80}))
				transitionTo(GamePhase::Learning);
		}

		void updateGameOverPhase()
		{
			drawGameBackground();
			m_enemy.draw();
			m_bulletCurtain.draw();
			m_effect.update();

			Rect{0, 0, Config::WindowSize}.draw(ColorF{0, 0, 0, 0.5});

			drawGameOverUI();
		}

		void drawGameOverUI()
		{
			m_font(U"Score: {}"_fmt(m_score)).draw(20, 20, Palette::White);
			m_font(U"ゲームオーバー").drawAt(400, 100, Palette::White);
			m_font(U"Final Score").drawAt(400, 180, Palette::Yellow);
			m_font(U"{}"_fmt(m_score)).drawAt(400, 260, Palette::White);

			const Rect tweetBtn{Arg::center(400, 380), 240, 60};
			tweetBtn.rounded(10).draw(tweetBtn.mouseOver() ? ColorF{0.1, 0.6, 1.0} : ColorF{0.0, 0.5, 0.9});
			m_smallFont(U"Twitterで共有").drawAt(tweetBtn.center(), Palette::White);

			if (tweetBtn.leftClicked())
			{
				const String tweetText = U"VowelShooting で {} 点獲得しました！\n母音操作シューティングゲーム (MFCC を Siv3D に組み込むサンプル) #VowelShooting"_fmt(m_score);
				Twitter::OpenTweetWindow(tweetText);
			}

			const Rect restartBtn{Arg::center(400, 470), 240, 60};
			restartBtn.rounded(10).draw(restartBtn.mouseOver() ? ColorF{0.2, 0.7, 0.2} : ColorF{0.1, 0.5, 0.1});
			m_smallFont(U"リスタート").drawAt(restartBtn.center(), Palette::White);

			if (restartBtn.leftClicked())
				transitionTo(GamePhase::Playing);

			if (SimpleGUI::Button(U"再学習", Vec2{20, 80}))
				transitionTo(GamePhase::Learning);
		}
	};
}

void Main()
{
	Window::SetTitle(U"Voice Controller Danmaku");
	Window::Resize(GameSystem::Config::WindowSize);

	AudioCore::MFCCAnalyzer mfccAnalyzer;
	Microphone mic{StartImmediately::Yes};
	UserInterface::AppUI appUI;

	while (System::Update())
	{
		const bool micDisconnected = System::EnumerateMicrophones().none([&](const auto& info)
		{
			return info.microphoneIndex == mic.microphoneIndex();
		});

		if (micDisconnected)
			mic.open(StartImmediately::Yes);

		appUI.update(mic, mfccAnalyzer);
	}
}
