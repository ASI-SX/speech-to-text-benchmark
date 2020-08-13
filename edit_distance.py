import editdistance

ref_text =
cheetah_text =
leopard_text =

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--engine_type', type=str, required=True)
    args = parser.parse_args()

    dataset = Dataset.create('librispeech')
    print('loaded %s with %.2f hours of data' % (str(dataset), dataset.size_hours()))

    engine = ASREngine.create(ASREngines[args.engine_type])
    print('created %s engine' % str(engine))

    word_error_count = 0
    word_count = 0

    bar = progressbar.ProgressBar(maxval=dataset.size(),
                                  widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    bar.start()

    def normalize_nums(value):
        p = inflect.engine()
        for ind, word in enumerate(value, start=0):
            if any(c.isdigit() for c in word):
                # replace number in v with string
                # print(word)
                value[ind] = p.number_to_words(word)
        return value

    for i in range(dataset.size()):
        path, ref_transcript = dataset.get(i)

        transcript = engine.transcribe(path)
        # ref_transcript = ref_transcript.translate(str.maketrans('', '', string.punctuation))

        ref_words = ref_transcript.strip('\n ').lower().split()
        words = transcript.strip('\n ').lower().split()

        words = normalize_nums(words)

        if set(ref_words) - set(words):
            print("REF: " + ref_transcript)
            print("RESULT: " + transcript)
            print(set(ref_words)-set(words))

        word_error_count += editdistance.eval(ref_words, words)
        word_count += len(ref_words)

        bar.update(i + 1)

    print('word error rate : %.2f' % (100 * float(word_error_count) / word_count))
