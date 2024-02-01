import pandas as pd

df = pd.read_csv('data.csv', sep=',')
# keep the loc, gpe, camp, ghetto and O tags only
tag_list = ['B-GPE',' I-GPE', 'B-LOC','I-LOC','B-CAMP', 'I-CAMP', 'B-GHETTO', 'I-GHETTO']
df['new_tag_set'] = df['Tag'].apply(lambda x: x if x in tag_list else 'O')

#
# Identify sentence boundaries based on the full stop in the 'words' column
df['sentence_boundary'] = df['Gold'].str.endswith('.')
df['sentence_id'] = (df['sentence_boundary'].shift(fill_value=False) & ~df['sentence_boundary']).cumsum()
grouped_df = df.groupby('sentence_id').agg({'Gold': list, 'new_tag_set': list}).reset_index()
grouped_df['sentence'] = grouped_df['Gold'].apply(' '.join)

grouped_df['tags'] = grouped_df['new_tag_set'].apply(lambda x: [tag for tag in x if tag != '.'])
result_df = grouped_df[['Gold', 'new_tag_set', 'sentence']]

print(result_df)

# Function to extract bigrams from a sentence
def extract_bigrams(sentence):
    words = sentence.split()
    return [tuple(words[i:i + 2]) for i in range(len(words) - 1)]

# Specify the target word and window size
target_word = 'Auschwitz'
window_size = 2
# Insert a new sentence
new_sentence = "THe was taken to auschwitz camp in Poland."
new_bigrams = extract_bigrams(new_sentence)

# Extract bigrams from existing sentences and compare with the new sentence
# Extract bigrams from existing sentences and compare with the new sentence
similarities = []
for i, existing_sentence in result_df.iterrows():
    existing_words = existing_sentence['Gold']
    existing_tags = existing_sentence['new_tag_set']
    common_bigrams = 0
    for word, tag in zip(existing_words, existing_tags):
        if target_word in existing_sentence['sentence']:
            common_bigrams += int(word.lower() == target_word.lower())  # Increment count if the word matches the target
    similarities.append((i, common_bigrams))

# Sort and get the top 5 similar sentences based on common bigrams
top_similar_sentences = sorted(similarities, key=lambda x: x[1], reverse=True)[:5]

# Print the result
print(f"\nNew Sentence:\n{new_sentence}\n")
print("Top 5 Similar Sentences:")
for index, common_bigrams in top_similar_sentences:
    similar_sentence = result_df['sentence'][index]
    similar_tags = result_df['new_tag_set'][index]
    words_tags = [f"{word}-{tag}" for word, tag in zip(similar_sentence.split(), similar_tags)]
    print(" ".join(words_tags))
    print(f"Common Words with Target: {common_bigrams}")
    print()

#extract the top 5 sentences and their tags from the inital dataset
