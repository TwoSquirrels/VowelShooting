#include <Siv3D.hpp> // Siv3D v0.6.16

// --- MFCC 関連クラス群 ---

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

		// バッファのコピー.
		for (size_t pos : step(f.size()))
		{
			const size_t idx = (pos + writePos < f.size() ? mic.getBufferLength() : 0) + pos + writePos - f.size();
			f[pos] = buffer[idx].left;
		}

		// プリエンファシス (double -> float 警告回避のため明示的キャストとリテラルを使用).
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

		// bin 計算 (double -> size_t への変換警告を回避).
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

// --- ゲーム用ロジック ---

// 音量計算用ユーティリティ.
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
		// コマンドに応じて状態を変更.
		if (command == U"あ")
		{
			color = Palette::Red;
		}
		else
		{
			// "あ" 以外なら元の色に戻すなどの処理が必要であればここに記述.
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

		// 画面外に出ないように制限.
		pos = pos.clamp(Scene::Rect().stretched(-20));
	}

	void draw() const
	{
		Circle{pos, 30}.draw(color);
	}
};

void Main()
{
	Window::SetTitle(U"Voice Controller Game");
	Window::Resize(800, 600);

	MFCCAnalyzer mfccAnalyzer{};
	Microphone mic{StartImmediately::Yes};

	// 音量閾値 (これより小さい音は無視する).
	const double rmsThreshold = volumeToRMS(0.5);

	// 母音のラベルと学習データ.
	const Array<String> vowels = {U"あ", U"い", U"う", U"え", U"お"};
	Array<MFCC> learnedMFCCs(5);

	size_t currentLearningIndex = 0; // 現在学習中の母音インデックス.
	bool isGameMode = false; // ゲームモードかどうか.

	Player player{Scene::Center(), Palette::Skyblue};
	Font font{40};

	while (System::Update())
	{
		// マイクの初期化チェック.
		if (System::EnumerateMicrophones().none([&](const auto& info)
		{
			return info.microphoneIndex == mic.microphoneIndex();
		}))
		{
			mic.open(StartImmediately::Yes);
		}

		// 現在の音声解析.
		const auto mfcc = mfccAnalyzer.analyze(mic);
		const double currentRMS = mic.rootMeanSquare();

		// モード分岐.
		if (not isGameMode)
		{
			// --- 学習フェーズ ---
			Scene::SetBackground(Palette::Darkgray);

			if (currentLearningIndex < vowels.size())
			{
				const String target = vowels[currentLearningIndex];
				font(U"学習モード: 「{}」と言いながら\nSPACEキーを押し続けてください"_fmt(target))
					.drawAt(Scene::Center(), Palette::White);

				// スペースキーを押している間、学習データを記録し続ける (平均化はせず、最新の有効なフレームを採用する簡易実装).
				if (KeySpace.pressed() && currentRMS > rmsThreshold && !mfcc.isUnset())
				{
					learnedMFCCs[currentLearningIndex] = mfcc;

					// 入力が十分にあれば進捗バーを表示.
					Rect{0, 500, Scene::Width(), 20}.draw(Palette::Gray);
					Rect{0, 500, static_cast<int>(Scene::Width() * (mic.rootMeanSquare() / 0.5)), 20}.draw(Palette::Orange);
				}

				// キーを離した瞬間に次の母音へ.
				if (KeySpace.up())
				{
					if (!learnedMFCCs[currentLearningIndex].isUnset())
					{
						currentLearningIndex++;
					}
				}
			}
			else
			{
				// 学習完了.
				font(U"学習完了！\nENTERキーでゲーム開始").drawAt(Scene::Center(), Palette::Yellow);
				if (KeyEnter.down())
				{
					isGameMode = true;
				}
			}
		}
		else
		{
			// --- ゲームフェーズ ---
			Scene::SetBackground(Palette::White);
			font(U"声で操作してください").drawAt(400, 50, Palette::Black);

			// 現在の音声が閾値を超えている場合のみ判定.
			String detectedVowel = U"";
			double maxSimilarity = 0.0;

			if (currentRMS > rmsThreshold && !mfcc.isUnset())
			{
				int bestIndex = -1;

				// 学習データと比較して最も似ている母音を探す.
				for (size_t i : step(vowels.size()))
				{
					double similarity = mfcc.cosineSimilarity(learnedMFCCs[i]);
					if (similarity > maxSimilarity)
					{
						maxSimilarity = similarity;
						bestIndex = static_cast<int>(i);
					}
				}

				// 類似度が一定以上ならコマンドとして採用.
				if (bestIndex != -1 && maxSimilarity > 0.85) // 0.85は判定の厳しさ.
				{
					detectedVowel = vowels[bestIndex];
				}
			}

			// プレイヤーの更新と描画.
			player.update(detectedVowel, Scene::DeltaTime());
			player.draw();

			// デバッグ表示: 現在の認識結果.
			if (detectedVowel)
			{
				font(detectedVowel).drawAt(Scene::Center().movedBy(0, -100), Palette::Black);
			}

			// 操作ガイド.
			font(U"あ:赤  い:←  う:↑  え:→  お:↓").drawAt(400, 550, Palette::Gray);
		}
	}
}
