"""
Author: Dario Pittera
Contact: www.dariopittera.com
Date: 25/02/2024

"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# NLP libraries
import nltk
from nltk.corpus import stopwords
from nltk.util import ngrams
nltk.download('stopwords')
italian_stopwords = stopwords.words('italian')
english_stopwords = stopwords.words('english')
from PIL import Image

# for n-grams
from sklearn.feature_extraction.text import CountVectorizer

# to summarise text
from transformers import pipeline

## for stop words
from wordcloud import WordCloud

# sentiment analysis for Italian language
# from sentita import calculate_polarity # need keras 2.15.0 - don't install tensorflow

st.set_option('deprecation.showPyplotGlobalUse', False) # enable/disable warnings

@st.cache_data
def plotFromData(csv):

    try:
        title = csv.split(".")[0]
    except: 
        title = csv.name.split(".")[0]
        
    # import data
    df = pd.read_csv(csv, encoding='latin-1', skiprows=0, index_col=0)
    df.dropna(subset=["comment"], inplace=True)
    
    # take all the comments
    text = " ".join(review for review in df.comment)
    print ("There are {} words in the combination of all review.".format(len(text)))
      
    # Create stopword list:
    stopwords = set(italian_stopwords)
    stopwords.update(["canzone", "song", "molto","sanremo","festival",title,title.lower()])
    stopwords.update(english_stopwords)

    # Generate a word cloud image
    plt.figure(figsize=(15,8))
    wordcloud = WordCloud(stopwords=stopwords, background_color="white", width=1600, height=800).generate(text)

    # Display the generated image:
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    st.pyplot(plt.gcf())

def get_top_ngram(corpus, n=None):
   
    vec = CountVectorizer(ngram_range=(n, n)).fit(corpus)
    bag_of_words = vec.transform(corpus)  
    sum_words = bag_of_words.sum(axis=0) 
    
    words_freq = [(word, sum_words[0, idx]) 
                  for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:10]

def newPar(num=1):
    for n in range(0,num):
        st.write("")

def chooseMood(mood_value):
    match mood_value:
        case _ if mood_value <= 0.2:
            mood_name = "anger"
        case _ if 0.2 < mood_value <= 0.4:
            mood_name = "smile"
        case _ if 0.4 < mood_value <= 0.5:
            mood_name = "smile"
        case _ if mood_value > 0.5:
            mood_name = "happy"   
    return mood_name

#############################################
############# BEGINNING #####################
#############################################

st.set_page_config(
    page_title="NLP on Sanremo 2024 singers",
    page_icon="🎤",
    layout="wide",
)

st.title("NLP on Sanremo 2024 singers")

st.header("Example 1")

st.text("")
st.write("Let's start with Angelina Mango, singing his father's cover 'La rondine'. Below, \
          is an extract from the data set containing the comments we are going to analyse. \
          Clearly, the comments are mostly in Italian language. His father was loved in \
          Italy and this shows through the comments. We can also see that there are a few \
          words that could be eliminated by setting some stopwords. We will do it in example 3. \
          Below, is an extract of the comments we are going to plot on the word oud.")

# import data
df = pd.read_csv("angelina_lower.csv", encoding='latin', skiprows=0, index_col=0)
df.dropna(subset=["comment"], inplace=True)
st.write(df.head(5))

plotFromData("angelina_lower.csv")

##########################################################################
##########################################################################

st.header("Example 2 - upload your file")

st.text("")
st.write("Now let's repeat the process with a different video. Please, upload your csv containing new comments.")

uploaded_file = st.file_uploader(label="Upload your comments file")
if uploaded_file is not None:
    plotFromData(uploaded_file)

#################################################### DASH

st.header("Example 3 - Simple dashboard")

st.write("Now, throught the Google APIs we collected 400 comments for each of the contestant of the Festival \
          of Sanremo 2024. We will now build a dashboard to show the comments' juice at a glance.\
          Specifically, we show a wordcloud, a variable n-grams, the language of the comment and \
          sentiment analysis of the comments. Finally, we will show the most salient comments \
          - I won't take responsibility for what will be shown 😅.")

# import new df
df = pd.read_csv("singers.csv", encoding='latin', skiprows=0, index_col=0)
df.dropna(subset=["comment"], inplace=True)
df["singer"] = [x.title() for x in df["singer"]]

# top-level filters
singer_filter = st.selectbox("Select singer", sorted(pd.unique(df["singer"])))

# dataframe filter
df = df[df["singer"] == singer_filter]

# First row with two columns
col1, col2 = st.columns(2)

with col1:
    newPar()

    st.markdown("#### Word cloud")
    new_stopwords = st.text_input("Insert additional stopwords, comma space separated:")
    new_stopwords = new_stopwords.split(", ")
    singers_names = [x for x in df["singer"].iloc[0].split()] # remove singer's name from words cloud

    # take all the comments and just count words in our comments
    text = " ".join(review for review in df.comment)
    st.text("The comments considered total to {} words.".format(len(text)))

    # START PLOTTING
    # Create stopword list:
    stopwords = set(italian_stopwords)
    stopwords.update(english_stopwords)
    stopwords.update(["canzone", "canzoni", "song", "songs", "festival", "sanremo", "br", \
                      "www", "youtube", "sempre", "mai", "comunque", "quot", "minuto", \
                      "minuti", df["singer"].iloc[0],df["singer"].iloc[0].lower(), *new_stopwords, *singers_names
                      ])

    # Generate a word cloud image
    wordcloud = WordCloud(stopwords=stopwords, background_color="white", width=800, height=400).generate(text)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    # plt.margins(x=0,y=0)
    st.pyplot(plt.gcf(), clear_figure=True)

with col2:
    newPar()
    
    st.markdown("#### N-gram")

    how_many_together = st.number_input("No of n-grams:", min_value=0, max_value=10, value=4, step=1, on_change=None)
    
    top_n_bigrams = get_top_ngram(df.comment,how_many_together)[:10]     
    top_n_bigrams = [gram for gram in top_n_bigrams if not any(st in gram for st in stopwords)]
    
    x,y=map(list,zip(*top_n_bigrams)) 
    sns.barplot(x=y,y=x)
    plt.title(f"{how_many_together}-gram")    
    st.pyplot(plt.gcf(), clear_figure=True)

############################################
# Second row with a single centered column #
############################################

col3, col4 = st.columns(2)

with col3:
    # languageDetection(df)
    st.markdown("#### Comments from the world")

    nations_label = list(df[df['singer']==singer_filter]['lang'].value_counts().index)
    nations_count = list(df[df['singer']==singer_filter]['lang'].value_counts())

    # DONUT PLOT
    # The slices will be ordered and plotted counter-clockwise.
    labels = nations_label
    sizes = nations_count
    explode = np.zeros(len(nations_label))
    explode[0] = 0.04

    _, texts, autotexts = plt.pie(sizes, labels=labels, pctdistance=0.75, labeldistance=1.1, explode=explode,
            startangle=0, autopct='%i%%', shadow=False, rotatelabels=False)

    for text in texts:
        text.set_color('black')
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_size(11)
        autotext.set_weight("bold")
            
    # Draw a circle at the pie's centre
    centre_circle = plt.Circle((0,0),0.55,color='white', fc='white',linewidth=1.25)
    fig = plt.gcf()
    fig.gca().add_artist(centre_circle)

    # Set aspect ratio to be equal to draw a circular shape
    plt.axis('equal')
    st.pyplot(plt.gcf(), clear_figure=True)

with col4:

    # st.markdown("#### Sentiment Analysis - this might take a while")  
    # st.write('Angry icons created by Eucalyp - [Flaticons](https://www.flaticon.com/free-icons/angry)')

    # query = df.loc[(df["singer"] == singer_filter) & (df["lang"] == "it")]["comment"]
    # results, polarities = calculate_polarity(query)
    # emo_res = pd.DataFrame(polarities, columns=['pos', 'neg'])

    # # check the overall mood and select the correct image
    # mood = emo_res["pos"].mean() - emo_res["neg"].mean()                
    # mood_name = chooseMood(mood)
    # im = Image.open(f'./emoji/{mood_name}.png')
    # im.thumbnail((150, 150))

    # # Convert relative positions to figure coordinates
    # fig = plt.gcf()
    # ax = plt.gca()
    # x_fig, y_fig = ax.transAxes.transform([1.65, 1.65])


    # sns.barplot(emo_res, palette=["#77DD77", "#FF6961"])
    # plt.title(f"Overall comments' mood for {df['singer'].iloc[0]}");
    # plt.xlabel("Valence")
    # plt.ylabel("Sentiment strength")
    # plt.ylim(0,1)
    # # plt.figimage(im, xo=460, yo=330)
    # plt.figimage(im, x_fig, y_fig)
    # st.pyplot(plt.gcf(), clear_figure=True)
    pass


##################################

st.markdown("#### Short summary - this might take a while")  

# Load the summarization pipeline
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Example text to summarize
text = " ".join(review for review in df[df["singer"]==singer_filter]["comment"])

# Truncate the input text to the maximum supported length
max_input_length = summarizer.model.config.max_position_embeddings
truncated_text = text[:max_input_length]

# Summarize the text
summary = summarizer(truncated_text, max_length=150, min_length=50, do_sample=False)

# Print the summary
st.write(summary[0]['summary_text'])
    