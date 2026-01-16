from textblob import TextBlob

text = "The deployment failed and everything broke"

analysis = TextBlob(text)

print("Text:", text)
print("Sentiment score:", analysis.sentiment.polarity) # type: ignore

if analysis.sentiment.polarity > 0: # type: ignore
    print("Overall sentiment: Positive 🙂")
elif analysis.sentiment.polarity < 0: # type: ignore
    print("Overall sentiment: Negative 🙁")
else:
    print("Overall sentiment: Neutral 😐")
