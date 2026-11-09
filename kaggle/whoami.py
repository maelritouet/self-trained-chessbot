"""Print the authenticated Kaggle username (and nothing else, never the token)."""

from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi()
api.authenticate()
print(api.get_config_value("username"))
