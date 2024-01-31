import pandas as pd
import spacy

# Load spaCy model
nlp = spacy.load('en_core_web_sm')

# Example DataFrame
df = pd.read_csv('data.csv', sep=',')
# keep the loc, gpe, camp, ghetto and O tags only
tag_list = ['B-GPE',' I-GPE', 'B-LOC','I-LOC','B-CAMP', 'I-CAMP', 'B-GHETTO', 'I-GHETTO']
df['new_tag_set'] = df['Tag'].apply(lambda x: x if x in tag_list else 'O')


# Identify sentence boundaries based on the full stop in the 'Gold' column
df['sentence_boundary'] = df['Gold'].str.endswith('.')
df['sentence_id'] = (df['sentence_boundary'].shift(fill_value=False) & ~df['sentence_boundary']).cumsum()
grouped_df = df.groupby('sentence_id').agg({'Gold': list, 'Tag': list}).reset_index()
grouped_df['sentence'] = grouped_df['Gold'].apply(' '.join)

grouped_df['new_tag_set'] = grouped_df['Tag'].apply(lambda x: x if x in ['B-GPE','I-GPE','B-LOC','I-LOC','B-CAMP','I-CAMP','B-GHETTO','I-GHETTO'] else 'O')
grouped_df['tags'] = grouped_df['new_tag_set'].apply(lambda x: [tag for tag in x if tag != '.'])
result_df = grouped_df[['Gold', 'new_tag_set', 'sentence']]

print(result_df)

# Function to extract POS tags from a sentence using spaCy
def extract_pos_tags(sentence):
    doc = nlp(sentence)
    return [token.text + "/" + token.pos_ for token in doc]

# Specify the target word and window size
target_word = 'Auschwitz'
window_size = 2

# Insert a new sentence
new_sentence = "He was taken to Auschwitz camp in Poland."

new_pos_tags = extract_pos_tags(new_sentence)

# Extract POS tags from existing sentences and compare with the new sentence
similarities = []
for i, existing_sentence in result_df.iterrows():
    existing_pos_tags = extract_pos_tags(existing_sentence['sentence'])
    common_pos_tags = len(set(new_pos_tags).intersection(existing_pos_tags))
    if target_word in existing_sentence['sentence']:
        similarities.append((existing_sentence['sentence'], common_pos_tags))


top_similar_sentences = sorted(similarities, key=lambda x: x[1], reverse=True)[:5]

filtered_top_similar_sentences = [(sentence, common_pos_tags) for sentence, common_pos_tags in top_similar_sentences if target_word in sentence]

# Print the result
print(f"\nNew Sentence:\n{new_sentence}\n")
print("Top 5 Similar Sentences:")
for sentence, common_pos_tags in filtered_top_similar_sentences:
    print(f"Sentence: {sentence}")
    print(f"POS Tags for Given Sentence: {new_pos_tags}")
    print(f"POS Tags for Predicted Sentence: {extract_pos_tags(sentence)}")
    print(f"Common POS Tags: {common_pos_tags}\n")
