import pandas as pd

df = pd.read_csv('data.csv', sep=',')
# keep the loc, gpe, camp, ghetto and O tags only 
tag_list = ['B-GPE', ' I-GPE', 'B-LOC', 'I-LOC', 'B-CAMP', 'I-CAMP', 'B-GHETTO', 'I-GHETTO']
df['new_tag_set'] = df['Tag'].apply(lambda x: x if x in tag_list else 'O')

# Identify sentence boundaries based on the full stop in the 'words' column
df['sentence_boundary'] = df['Gold'].str.endswith('.')
df['sentence_id'] = (df['sentence_boundary'].shift(fill_value=False) & ~df['sentence_boundary']).cumsum()
grouped_df = df.groupby('sentence_id').agg({'Gold': list, 'new_tag_set': list}).reset_index()
grouped_df['tags'] = grouped_df['new_tag_set'].apply(lambda x: [tag for tag in x if tag != '.'])
grouped_df['sentence'] = grouped_df['Gold'].apply(' '.join)
result_df = grouped_df[['Gold', 'new_tag_set', 'sentence']]

# Print the result for one instances
desired_row_index = 0
desired_row = result_df.iloc[desired_row_index]
print(desired_row)


# extract the window size relavant to the target word.
def extract_window_around_target(text, target_word, window_size):
    words = text.split()
    try:
        target_index = words.index(target_word)
        start_index = max(0, target_index - window_size)
        end_index = min(len(words), target_index + window_size + 1)
        window_words = words[start_index:end_index]
        target_index_in_window = target_index - start_index

        # If we want to measure the before and after parts of the code
        begin_words = window_words[:target_index_in_window]
        after_words = window_words[target_index_in_window + 1:]

        bigrams_before = [window_words[i:i + 2] for i in range(target_index_in_window) if
                          i + 1 < target_index_in_window]
        bigrams_after = [window_words[i:i + 2] for i in range(target_index_in_window + 1, len(window_words) - 1) if
                         i + 1 < len(window_words) - 1]

        return bigrams_before, bigrams_after

    except ValueError:
        return None


# Example usage:
# input_text = "This is an example text to demonstrate window extraction in Python."
# for i in result_df['sentence']:
#     input_text = i
#     target_word = "living"
#     window_size = 4
#
#     bigrams_before, bigrams_after = extract_window_around_target(input_text, target_word, window_size)
#
#     print("Bigrams before:", bigrams_before)
#     print("Bigrams after:", bigrams_after)

target_word = "living"
window_size = 4

result_df['windows'] = result_df['sentence'].apply(lambda x: extract_window_around_target(x, target_word, window_size))
result_df = result_df.dropna().reset_index(drop=True)

# Print the result
print(result_df[['sentence', 'windows']])


