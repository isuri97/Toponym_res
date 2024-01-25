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

# Function to extract bigrams from a sentence
def extract_bigrams(sentence):
    words = sentence.split()
    return [tuple(words[i:i+2]) for i in range(len(words)-1)]

# Function to filter sentences in result_df based on the target word
def filter_sentences_by_target(result_df, target_word):
    return result_df[result_df['sentence'].str.contains(target_word, case=False)]

# Function to adjust result_df according to the given window size
def adjust_result_df(result_df, target_word, window_size):
    adjusted_sentences = []
    for i, sentence in result_df.iterrows():
        if target_word in sentence['sentence']:
            words_in_window = sentence['sentence'].split()
            target_index = words_in_window.index(target_word)
            start_index = max(0, target_index - window_size)
            end_index = min(len(words_in_window), target_index + window_size + 1)
            window_words = words_in_window[start_index:end_index]
            adjusted_sentence = ' '.join(window_words)
            adjusted_sentences.append(adjusted_sentence)
    return pd.DataFrame({'sentence': adjusted_sentences})

# Specify the target word and window size
target_word = 'living'
window_size = 3

# Insert a new sentence
new_sentence = "This is a new living in the Germany."
new_bigrams = extract_bigrams(new_sentence)

# Filter result_df based on the target word
filtered_result_df = filter_sentences_by_target(result_df, target_word)

# Adjust result_df according to the given window size
adjusted_result_df = adjust_result_df(result_df, target_word, window_size)

# Extract bigrams from the adjusted result_df
adjusted_bigrams = []
for i, sentence in adjusted_result_df.iterrows():
    adjusted_bigrams.extend(extract_bigrams(sentence['sentence']))

# Calculate similarity with the new sentence
similarities = []
for i, sentence in filtered_result_df.iterrows():
    existing_bigrams = extract_bigrams(sentence['sentence'])
    common_bigrams = len(set(new_bigrams).intersection(existing_bigrams))
    similarities.append((i, common_bigrams))

# Sort and get the top 5 similar sentences based on common bigrams
top_similar_sentences = sorted(similarities, key=lambda x: x[1], reverse=True)[:5]


print(f"\nNew Sentence:\n{new_sentence}\n")
print(f"Filtered Result DataFrame (Containing '{target_word}'):")
print(filtered_result_df)
print(f"\nAdjusted Result DataFrame (Window Size {window_size}):")
print(adjusted_result_df)
print(f"\nBigrams from Adjusted Result DataFrame:")
print(adjusted_bigrams)
print(f"\nTop 5 Similar Sentences containing '{target_word}':")
for index, common_bigrams in top_similar_sentences:
    print(f"{filtered_result_df['sentence'][index]} (Common Bigrams: {common_bigrams})")
