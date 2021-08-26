import random
from run_generation import Generation
from nltk import tokenize


class DocumentAugmentation():
    """
    Document Augmentation Approaches
    """

    def __init__(self, n, input):
        """
        Initialize the class
        :param n: how many sentences are selected for augmentation
        :param input: input sequence, string
        """
        self.n = n
        self.input = input
        self.sentences = tokenize.sent_tokenize(input)

    def RandomInsertion(self, fp16, num_sent_context=3, model='gpt2'):
        """
        randomly insert a sentence in the input document, which is generated based on its context. Do this n times.
        :return:
        """
        self.augmented_sentences = self.sentences
        generation = Generation(model_type=model, fp16=fp16)

        for i in range(self.n):
            # generate the index for inserting
            location = random.randrange(len(self.sentences))

            before_idx_start, before_idx_end = location - num_sent_context, location
            after_idx_start, after_idx_end = location + 1, location + num_sent_context + 1

            context = " ".join(
                self.augmented_sentences[max(0, before_idx_start): max(0, before_idx_end)]) + " ".join(
                self.augmented_sentences[
                min(len(self.augmented_sentences) - 1, after_idx_start): min(len(self.augmented_sentences) - 1,
                                                                             after_idx_end)])

            new_sentence = generation.generate(context)

            if new_sentence[-1] not in ["?", ".", "!"]:
                new_sentence += "."

            # insert new_sentence into the self.sentences
            if location < 0:
                self.augmented_sentences = [new_sentence] + self.sentences
            else:
                update_sentence = self.sentences[:location + 1]
                update_sentence.append(new_sentence)
                update_sentence += self.sentences[location + 1:]
                self.augmented_sentences = update_sentence

    def RandomSwap(self):
        """
        randomly select two sentences in the input document and swap their positions. Do this $n$ times.
        :return:
        """
        self.augmented_sentences = self.sentences
        if len(self.sentences) >= 2:
            for i in range(self.n):
                # location is a list contains two random numbers selected
                location = random.sample(range(len(self.augmented_sentences)), 2)
                sent1 = self.augmented_sentences[location[0]]
                sent2 = self.augmented_sentences[location[1]]
                # swap two sentences
                self.augmented_sentences[location[0]], self.augmented_sentences[location[1]] = sent2, sent1

    def RandomDeletion(self):
        """
        randomly delete n sentences from the input document.
        :return:
        """
        self.augmented_sentences = self.sentences
        # Here we require that the augmented document should have at least one sentence
        if self.n <= len(self.sentences) - 1:
            # location is a list contains two random numbers selected
            location_delete = random.sample(range(len(self.augmented_sentences)), self.n)
            update_sentence = [self.augmented_sentences[i] for i in range(len(self.augmented_sentences)) if
                               i not in location_delete]
            self.augmented_sentences = update_sentence

    def LanguageGenerationReplacement(self, fp16, model="gpt2", num_sent_context=3):
        """
        randomly choose n sentences from the input document.
        Replace each of these sentences with a newly generated sentence based on its context.
        :return:
        """
        self.augmented_sentences = self.sentences
        generation = Generation(model_type=model, fp16=fp16)

        if self.n <= len(self.sentences):
            location = random.sample(range(len(self.augmented_sentences)), self.n)
            update_sentence = []
            for i in range(len(self.augmented_sentences)):
                if i not in location:
                    update_sentence.append(self.augmented_sentences[i])
                else:
                    before_idx_start, before_idx_end = i - num_sent_context, i
                    after_idx_start, after_idx_end = i + 1, i + num_sent_context + 1

                    context = " ".join(
                        self.augmented_sentences[max(0, before_idx_start): max(0, before_idx_end)]) + " ".join(
                        self.augmented_sentences[
                        min(len(self.augmented_sentences) - 1, after_idx_start): min(len(self.augmented_sentences) - 1,
                                                                                     after_idx_end)])
                    new_sentence = generation.generate(context)
                    update_sentence.append(new_sentence)
            self.augmented_sentences = update_sentence

    def DocumentRotation(self):
        """
        randomly select a sentence and rotate the document using this sentence. Do this n times.
        :return:
        """
        self.augmented_sentences = self.sentences

        # perform experiments n times
        for i in range(self.n):
            # generate the index for the selected sentence
            location = random.randrange(len(self.sentences))
            # rotate the document
            self.augmented_sentences = self.augmented_sentences[location + 1:][::-1] + [
                self.augmented_sentences[location]] + self.augmented_sentences[:location][::-1]

    def RandomInsertionFromDoc(self):
        """
        Simplified version of RandomInsertion:
        randomly insert n sentence in the input document, which is selected from document itself.
        :return:
        """
        self.augmented_sentences = self.sentences

        for i in range(self.n):
            # generate the index for inserting
            location = random.randrange(len(self.sentences))

            # randomly select a sentence from the input document
            select_location = random.randrange(len(self.sentences))
            new_sentence = self.sentences[select_location]

            update_sentence = self.sentences[:location + 1]
            update_sentence.append(new_sentence)
            update_sentence += self.sentences[location + 1:]
            self.augmented_sentences = update_sentence
