import time
import random
import speech_recognition as sr

def read_sentences_from_file(sentences):
    """Read a list of sentences from a file."""
    with open(sentences, "r", encoding="utf-8") as f:
        sentences = [line.strip() for line in f.readlines()]
    return sentences

sentences = read_sentences_from_file("sentences.txt")
sentence = random.choice(sentences)

def perform_sentence(sentence, delay=0.2):
    """Simulate speaking a sentence and record the user repeating the sentence."""
    r = sr.Recognizer()
    r.energy_threshold = 200
    # Split the sentence into individual words
    words = sentence.split()
    # Prompt the user to speak the sentence
    print("Please speak the following sentence:")
    print(sentence)
    # Wait for the user to speak and recognize their speech
    with sr.Microphone() as source:
        audio = r.listen(source)
        text = ""
        try:
            text = r.recognize_google(audio)
            print("You said:", text)
            # Record the user repeating the sentence
            record_user_repeating(text)
            # Check each word to see if it was spoken correctly
            for i, word in enumerate(words):
                if i < len(text.split()):
                    if text.split()[i].lower() == word.lower():
                        print("\033[32m" + word + "\033[0m", end=" ")  # print in green
                    else:
                        print("\033[31m" + word + "\033[0m", end=" ")  # print in red
                        if not repeat_word(word):
                            break
                else:
                    print("\033[31m" + word + "\033[0m", end=" ")  # print in red
            print()
        except sr.UnknownValueError:
            print("Sorry, I didn't understand.")
        except sr.RequestError as e:
            print("Could not request results from Google Speech Recognition service; {0}".format(e))
        # Simulate stuttering by randomly inserting delays between letters
        for char in text:
            if random.random() < 0.2:
                time.sleep(delay)
            print(char, end="", flush=True)
            time.sleep(delay)
        print("\n")
def record_user_repeating(text):
    """Record the user repeating the sentence and save the recording to a file."""
    filename = "user_repetition.wav"
    print("Please repeat the sentence you just said:")
    r = sr.Recognizer()
    with sr.Microphone() as source:
        audio = r.listen(source)
        with open(filename, "wb") as f:
            f.write(audio.get_wav_data())

def repeat_word(word):
    """Prompt the user to repeat the incorrect word."""
    r = sr.Recognizer()
    while True:
        print("Please repeat the word:", word)
        with sr.Microphone() as source:
            audio = r.listen(source,timeout=10)
            try:
                text = r.recognize_google(audio)
                if text.strip().lower() == word.strip().lower():
                    print("Correct!")
                    # Play a success sound effect
                    # success_sound.play()
                    return True
                else:
                    print("Incorrect. Please try again.")
            except sr.UnknownValueError:
                print("Sorry, I didn't understand. Please try again.")
            except sr.RequestError as e:
                print("Could not request results from Google Speech Recognition service; {0}".format(e))

# Example usage
sentences = read_sentences_from_file("sentences.txt")
perform_sentence(sentence, delay=0.2)