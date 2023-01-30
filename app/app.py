import altair as alt
import streamlit as st
import streamlit.components.v1 as stc
import pandas as pd
import numpy as np
import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity,linear_kernel

pipe_lr  = joblib.load(open('models/emotion_classifier.pkl','rb'))
def predict_emotions(docx):
    results = pipe_lr.predict([docx])
    return results[0]

def get_predict_proba(docx):
    results = pipe_lr.predict_proba([docx])
    return results

def load_data(data):
    data = pd.read_csv(data)
    return data

def vectorize_to_cosine(data):
    cv = CountVectorizer()
    count_matrix = cv.fit_transform(data)
    cosine_sim = cosine_similarity(count_matrix)
    return cosine_sim

@st.cache
def recommend_courses(title,cosine_mat,df,num=5):
    course_indices = pd.Series(df.index,index=df['course_title']).drop_duplicates()
    idx = course_indices[title]
    sim_scores =list(enumerate(cosine_mat[idx]))
    sim_scores = sorted(sim_scores,key=lambda x: x[1],reverse=True)
    selected_course_indices = [i[0] for i in sim_scores[1:]]
    selected_course_scores = [i[0] for i in sim_scores[1:]]
    result_df = df.iloc[selected_course_indices]
    result_df['similarity_score'] = selected_course_scores
    final_recommended_courses = result_df[['course_title','similarity_score','url','price','num_subscribers']]
    return final_recommended_courses


df = load_data('data/udemy_course_data.csv')
@st.cache
def not_found(term):
    result = df[df['course_title'].str.contains(term)]
    return result

RESULT = """
<div style="width:90%;height:100%;margin:1px;padding:0px;position:relative;border-radius:5px;border-bottom-right-radius: 60px;
box-shadow:0 0 15px 5px #ccc; background-color: #a8f0c6;
  border-left: 5px solid #6c6c6c;">
<h4>{}</h4>
<p style="color:blue;"><span style="color:black;">📈Score::</span>{}</p>
<p style="color:blue;"><span style="color:black;">🔗</span><a href="{}",target="_blank">Link</a></p>
<p style="color:blue;"><span style="color:black;">💲Price:</span>{}</p>
<p style="color:blue;"><span style="color:black;">🧑‍🎓👨🏽‍🎓 Students:</span>{}</p>
</div>
"""
emoticons = {'anger':'😠' ,'boredom':'😪', 'empty': '🤔' , 'enthusiasm':'🤗', 'fun': '🤪','happiness': '😊','hate': '😤','love': '🥰' ,'neutral': '😐' ,'relief':'😌' ,'sadness':'😞' ,'surprise': '😲','worry': '😰' }
def main():
    menu = ['Emotion-detection App','Course Recommendation App']
    choice = st.sidebar.selectbox('Menu', menu)
    if choice=='Emotion-detection App':
        st.subheader('Emotion-detection App')
        with st.form(key='emotion_check'):
            raw_text = st.text_area('Enter Text')
            submit_text = st.form_submit_button(label='Submit')

        if submit_text:
            col1,col2 = st.columns(2)
            prediction = predict_emotions(raw_text)
            proba = get_predict_proba(raw_text)
            with col1:
                st.success('Original Text')
                st.write(raw_text)
                st.success("Prediction")
                emoji_icon = emoticons[prediction]
                st.write("{}:{}".format(prediction,emoji_icon))
                st.write("Confidence:{}".format(np.max(proba)))
            with col2:
                st.success('Prediction Probability')
                # st.write(proba)
                proba_df = pd.DataFrame(proba, columns=pipe_lr.classes_)
                # st.write(proba_df.T)
                proba_df_clean = proba_df.T.reset_index()
                proba_df_clean.columns = ['Emotion','Probability']

                fig = alt.Chart(proba_df_clean).mark_bar().encode(
                    x='Emotion',
                    y='Probability',
                    color = 'Emotion'
                )
                st.altair_chart(fig, use_container_width=True)
    else:
        st.subheader('Course Recommendation App')
        mat = vectorize_to_cosine(df['course_title'])
        search_term = st.text_input('Search')
        if st.button('Recommend'):
            if search_term is not None:
                try:
                    res = recommend_courses(search_term,mat,df)
                    with st.expander("Results as JSON"):
                        results_json = res.to_dict('index')
                        st.write(results_json)
                    for row in res.iterrows():
                        rec_title = row[1][0]
                        rec_desc = row[1][1]
                        rec_url = row[1][2]
                        rec_price = row[1][3]
                        rec_subscribers = row[1][4]

                        stc.html(RESULT.format(rec_title,rec_desc,rec_url,rec_price,rec_subscribers),height=200)
                except:
                    res = "Not Found"
                    st.warning(res)
                    st.info('Suggested Options:')
                    result = not_found(search_term)
                    # st.dataframe(result)
                    with st.expander("Results as JSON"):
                        results_json = result.to_dict('index')
                        st.write(results_json)
                    for row in result.iterrows():
                        rec_title = row[1][0]
                        rec_desc = row[1][1]
                        rec_url = row[1][2]
                        rec_price = row[1][3]
                        rec_subscribers = row[1][4]

                        stc.html(RESULT.format(rec_title,rec_desc,rec_url,rec_price,rec_subscribers),height=200)

                # st.write(res)
                







if __name__ == '__main__':
    main()