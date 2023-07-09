import speech_recognition as sr


def speech_recognition():
    # Create a recognizer instance
    r = sr.Recognizer()

    # Start listening to the microphone
    with sr.Microphone() as source:
        print("Listening....")
        audio = r.listen(source)
        try:
            # Recognize speech using Google Speech Recognition
            value = r.recognize_google(audio)
        except sr.UnknownValueError:
            value = "Google Speech Recognition could not understand audio"
        except sr.RequestError as e:
            value = "Could not request results from Google Speech Recognition service; {0}".format(
                e
            )

        return value
