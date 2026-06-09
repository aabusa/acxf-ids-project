from matplotlib.transforms import Transform
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler


#prepare the data for training

col_names = [
    "duration", "protocol_type", "service", "flag",
    "src_bytes", "dst_bytes", "land", "wrong_fragment",
    "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted",
    "num_root", "num_file_creations", "num_shells",
    "num_access_files", "num_outbound_cmds", "is_host_login",
    "is_guest_login", "count", "srv_count", "serror_rate",
    "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
    "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate",
    "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
    "label", "difficulty"
]

df = pd.read_csv("data/KDDTrain+.txt", names=col_names)
df.drop("difficulty", axis='columns', inplace=True)


protocol_encoder = LabelEncoder()
df["protocol_type"] = protocol_encoder.fit_transform(df["protocol_type"])
service_encoder = LabelEncoder()
df["service"] = service_encoder.fit_transform(df["service"])
flag_encoder = LabelEncoder()
df["flag"] = flag_encoder.fit_transform(df["flag"])

mapping = {
    "normal": "normal",
    "neptune": "dos",
    "back": "dos",
    "land": "dos",
    "pod": "dos",
    "smurf": "dos",
    "teardrop": "dos",
    "mailbomb": "dos",
    "apache2": "dos",
    "processtable": "dos",
    "udpstorm": "dos",
    "worm": "dos",
    "ipsweep": "probe",
    "nmap": "probe",
    "portsweep": "probe",
    "satan": "probe",
    "mscan": "probe",
    "saint": "probe",
    "ftp_write": "r2l",
    "guess_passwd": "r2l",
    "httptunnel": "r2l",
    "imap": "r2l",
    "multihop": "r2l",
    "named": "r2l",
    "phf": "r2l",
    "spy": "r2l",
    "sendmail": "r2l",
    "snmpgetattack": "r2l",
    "warezclient": "r2l",
    "warezmaster": "r2l",
    "snmpguess": "r2l",
    "xlock": "r2l",
    "xsnoop": "r2l",
    "buffer_overflow": "u2r",
    "loadmodule": "u2r",
    "perl": "u2r",
    "rootkit": "u2r",
    "sqlattack": "u2r",
    "xterm": "u2r",
    "ps": "u2r"

}
df["label"] = df["label"].map(mapping)
print(df["label"].value_counts())

# Separate features and labels

y = df["label"]
X = df.drop( "label", axis="columns")

print(X.shape)
print(y.shape)

# Normalize the features using Min-Max scaling
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
print(X_scaled)

# Encode the labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
print(y_encoded)

#y mapped to dos = 0, normal = 1, probe = 2, r2l = 3, u2r = 4 respectively
print(X_scaled.shape)
print(y_encoded.shape)
print(label_encoder.classes_)

# prepare the data for testing
test_df = pd.read_csv("data/KDDTest+.txt", names=col_names)
test_df.drop("difficulty", axis='columns', inplace=True)

# Encode the categorical features in the test set using the same label encoders used for the training set

test_df["protocol_type"] = protocol_encoder.transform(test_df["protocol_type"])
test_df["service"] = service_encoder.transform(test_df["service"])
test_df["flag"] = flag_encoder.transform(test_df["flag"])

# Map the attack labels in the test set to the same categories as in the training set
test_df["label"] = test_df["label"].map(mapping)

# Separate features and labels for the test set
y_test = test_df["label"]
X_test = test_df.drop("label", axis="columns")

# Normalize the features in the test set using the same scaler used for the training set
X_test_scaled = scaler.transform(X_test)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)

# Encode the labels in the test set using the same label encoder used for the training set
y_test_encoded = label_encoder.transform(y_test)

# Print the shapes of the preprocessed training and test sets and the distribution of labels in the training set
print(X_test_scaled.shape)
print(y_test_encoded.shape)
print(X_scaled.describe())
print(pd.Series(y_encoded).value_counts())

# Save the preprocessed data to .npy files
np.save("data/X_train.npy", X_scaled)
np.save("data/y_train.npy", y_encoded)
np.save("data/X_test.npy", X_test_scaled)
np.save("data/y_test.npy", y_test_encoded)