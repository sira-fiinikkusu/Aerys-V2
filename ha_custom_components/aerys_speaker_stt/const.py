"""Constants for Aerys Speaker STT."""
DOMAIN = "aerys_speaker_stt"
CONF_SOURCE_STT = "source_stt_entity"
CONF_RECOGNIZER_URL = "recognizer_url"
CONF_TIMEOUT = "recognizer_timeout"
DEFAULT_SOURCE_STT = "stt.home_assistant_cloud"
DEFAULT_RECOGNIZER_URL = "http://192.168.1.107:8099"
DEFAULT_TIMEOUT = 2.0
# hass.data[DOMAIN][LAST_KEY] = {"user_id", "confidence", "all_scores", "at"} — read by
# aerys_conversation within RESULT_TTL_S to tag the /ask body. Never a transcript prefix.
LAST_KEY = "last_result"
RESULT_TTL_S = 15.0
