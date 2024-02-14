from sklearn.ensemble import RandomForestClassifier
from asteroids import read_file

DEBUG = True

FEATURES_COMBINATIONS = [['a', 'e', 'n']]

POSITIVE_OBJECTS_FILE = 'cache/4J-2S-1.csv'
NEGATIVE_OBJECTS_FILE = 'cache/4J-2S-1-negatives.csv'
TEST_SET_FILE = 'cache/4J-2S-1-classifying.csv'

asteroid_numbers = read_file(POSITIVE_OBJECTS_FILE)

train_sizes_lst = []
for i in [50, 100]:
    train_sizes_lst.append({'positive': i, 'negative': asteroid_numbers[i] - i})

TRAIN_SIZES = train_sizes_lst

TEST_SIZES = [100000]

models_lst = {}
models_lst[f"RF d=0"] = RandomForestClassifier(
    random_state=42, n_estimators=100, max_depth=None, max_features='sqrt', min_samples_split=3, class_weight='balanced', n_jobs=-1
)

MODELS = models_lst
