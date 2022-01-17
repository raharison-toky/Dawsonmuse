from copyreg import pickle
from dawsonmuse import session_set
import pickle
with open( "sessions.p", "rb" ) as f:
	y = pickle.load(f)

def test_sessionset():
    event_dict = {"Congruent":2, "Incongruent":1}
    x = session_set(["recording1.csv","recording2.csv","recording3.csv","recording4.csv",],
    "mixed",event_dict,tmin=-0.2,tmax=1)
    assert x == y
    