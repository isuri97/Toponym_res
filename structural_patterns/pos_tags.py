import pandas as pd
import spacy

# Load spaCy model
nlp = spacy.load('en_core_web_sm')

# # Example DataFrame
# data = {'Gold': ['He', 'was', 'taken', 'to', 'Auschwitz', 'camp', 'in', 'Poland', '.'],
#         'Tag': ['O', 'O', 'O', 'O', 'B-CAMP', 'I-CAMP', 'O', 'B-GPE', 'O']}
# df = pd.DataFrame(data)

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
result_df = grouped_df[['Gold', 'new_tag_set', 'sentence', 'Tag']]  # Include 'Tag' column for original tags

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

# Extract POS tags for both the new and existing sentences
new_pos_tags = extract_pos_tags(new_sentence)

# Extract POS tags from existing sentences and compare with the new sentence
similarities = []
for i, existing_sentence in result_df.iterrows():
    existing_pos_tags = extract_pos_tags(existing_sentence['sentence'])
    # Count common POS tags
    common_pos_tags = len(set(new_pos_tags).intersection(existing_pos_tags))
    # Filter out sentences that don't contain the target word
    if target_word in existing_sentence['sentence']:
        similarities.append((existing_sentence['sentence'], common_pos_tags, existing_sentence['Tag']))

# Sort and get the top 5 similar sentences based on common POS tags
top_similar_sentences = sorted(similarities, key=lambda x: x[1], reverse=True)[:5]

# Print the result
print(f"\nNew Sentence:\n{new_sentence}\n")
print("Top 5 Similar Sentences:")
for sentence, common_pos_tags, tags in top_similar_sentences:
    predicted_pos_tags = extract_pos_tags(sentence)
    predicted_word_tag = " ".join(predicted_pos_tags)
    # print(f"POS Tags for Predicted Sentence: {predicted_word_tag}")
    original_word_tag = " ".join([f"{word}-{tag}" for word, tag in zip(sentence.split(), tags)])
    print(f"Original Tags for Sentence: {original_word_tag}")

    print(f"Common POS Tags: {common_pos_tags}\n")



