"""
Tests for simulate_gps_data
module
"""

import tempfile
import shutil
import sys
import time
import datetime
import requests
import pytest
import numpy as np
from forest.bonsai.simulate_gps_data import (
    get_basic_path,
    bounding_box,
    Person,
    gen_all_traj,
    process_attributes,
    sim_gps_data,
)
from forest.jasmine.data2mobmat import great_circle_dist
from forest.poplar.legacy.common_funcs import read_data


random_path = np.array(
    [
        [-2.59638, 51.458498],
        [-2.596069, 51.458726],
        [-2.595929, 51.458832],
        [-2.595808, 51.458777],
        [-2.595906, 51.458695],
        [-2.595986, 51.45861],
        [-2.596307, 51.458234],
        [-2.596366, 51.458222],
        [-2.596405, 51.458176],
        [-2.596418, 51.458159],
        [-2.596401, 51.458124],
        [-2.59648, 51.458046],
        [-2.596535, 51.457985],
        [-2.596569, 51.45795],
        [-2.596617, 51.457939],
        [-2.59664, 51.457913],
        [-2.596632, 51.457873],
        [-2.596701, 51.457797],
        [-2.597007, 51.45749],
        [-2.596998, 51.457463],
        [-2.597019, 51.457439],
        [-2.597031, 51.457425],
        [-2.597081, 51.457408],
        [-2.597129, 51.457365],
        [-2.597153, 51.457344],
        [-2.597137, 51.457307],
        [-2.597395, 51.457033],
        [-2.597428, 51.457003],
        [-2.597435, 51.456996],
        [-2.597508, 51.456975],
        [-2.597554, 51.456932],
        [-2.597647, 51.456976],
        [-2.597762, 51.457035],
        [-2.598061, 51.456853],
        [-2.598083, 51.456865],
        [-2.598104, 51.456877],
        [-2.597966, 51.456957],
        [-2.598148, 51.457153],
        [-2.598083, 51.457165],
        [-2.598191, 51.45726],
        [-2.59834, 51.45736],
        [-2.598432, 51.457402],
        [-2.598656, 51.457537],
        [-2.598695, 51.45756],
        [-2.598843, 51.457652],
        [-2.598883, 51.457629],
        [-2.598945, 51.457593],
        [-2.598973, 51.457578],
        [-2.599059, 51.457529],
        [-2.599326, 51.457704],
        [-2.599334, 51.457743],
        [-2.599372, 51.45775],
        [-2.599375, 51.457812],
        [-2.599687, 51.457821],
        [-2.599715, 51.457933],
        [-2.600055, 51.457987],
        [-2.600066, 51.458146],
        [-2.6001, 51.458197],
        [-2.600159, 51.458228],
        [-2.600336, 51.458319],
        [-2.60054, 51.45817],
        [-2.600724, 51.45811],
        [-2.601016, 51.458079],
        [-2.60109, 51.458081],
        [-2.60116, 51.458083],
        [-2.601341, 51.458086],
        [-2.60143, 51.458056],
        [-2.601617, 51.457993],
        [-2.602114, 51.457838],
        [-2.602177, 51.457821],
        [-2.602339, 51.45789],
        [-2.60244, 51.457986],
        [-2.602567, 51.458015],
        [-2.602877, 51.458222],
        [-2.602917, 51.458253],
        [-2.603008, 51.458312],
        [-2.603295, 51.45849],
        [-2.603532, 51.458551],
        [-2.603607, 51.458588],
        [-2.603631, 51.458604],
        [-2.603665, 51.458594],
        [-2.603672, 51.458608],
        [-2.60368, 51.458624],
        [-2.603692, 51.458649],
        [-2.603809, 51.458643],
        [-2.603826, 51.45868],
        [-2.603981, 51.458674],
        [-2.604272, 51.458688],
        [-2.604648, 51.458682],
        [-2.604761, 51.45868],
        [-2.604778, 51.45868],
        [-2.605122, 51.458668],
        [-2.605581, 51.458617],
        [-2.605745, 51.458598],
        [-2.606138, 51.458537],
        [-2.606402, 51.458474],
        [-2.606794, 51.45837],
        [-2.606939, 51.458293],
        [-2.607155, 51.458146],
        [-2.607177, 51.458124],
        [-2.60723, 51.458073],
        [-2.607405, 51.457954],
        [-2.607582, 51.457897],
        [-2.607792, 51.457827],
        [-2.608362, 51.457649],
        [-2.608464, 51.457639],
        [-2.608466, 51.457619],
    ]
)


def test_get_basic_path_simple_case():
    """
    Test simple case of getting basic path
    """
    basic_random_path = get_basic_path(random_path, "foot")
    boolean_matrix = basic_random_path == np.array(
        [
            [-2.59638, 51.458498],
            [-2.597031, 51.457425],
            [-2.598656, 51.457537],
            [-2.60109, 51.458081],
            [-2.603809, 51.458643],
            [-2.608466, 51.457619],
        ]
    )
    assert np.sum(boolean_matrix) == 12


def test_get_basic_path_small_path():
    """
    Test case of getting basic path from
    small path
    """
    random_path2 = random_path[-3:]
    basic_random_path2 = get_basic_path(random_path2, "bus")
    boolean_matrix2 = basic_random_path2 == random_path2
    assert np.sum(boolean_matrix2) == 6


def test_get_basic_path_compare_means():
    """
    Test lengths of getting basic path
    """
    basic_random_path_bicycle = get_basic_path(random_path, "bicycle")
    basic_random_path_car = get_basic_path(random_path, "car")
    basic_random_path_bus = get_basic_path(random_path, "bus")
    assert [
        len(basic_random_path_bicycle),
        len(basic_random_path_car),
        len(basic_random_path_bus),
    ] == [6, 4, 6]


rndm_coords = (51.458726, -2.596069)


def test_bounding_box_simple_case():
    """
    Test bounding box simple case
    """
    bbox = bounding_box(rndm_coords[0], rndm_coords[1], 500)
    actual_distance = np.round(
        great_circle_dist(rndm_coords[0], rndm_coords[1], bbox[0], bbox[1])
    )
    predicted_distance = np.round((2 * 500 ** 2) ** (1 / 2))
    assert actual_distance == predicted_distance


def test_bounding_box_zero_case():
    """
    Test case when 0 meters bounding box
    """
    bbox = bounding_box(rndm_coords[0], rndm_coords[1], 0)
    assert (bbox[0] == bbox[2]) and (bbox[1] == bbox[3])


@pytest.fixture(scope="session")
def all_nodes():
    """
    Create dictionary of locations of nodes
    """
    house_address = rndm_coords
    house_area = bounding_box(house_address[0], house_address[1], 2000)
    q_employment = ""
    q = """
        [out:json];
        (
            node{0}["amenity"="cafe"];
            node{0}["amenity"="bar"];
            node{0}["amenity"="restaurant"];
            node{0}["amenity"="cinema"];
            node{0}["leisure"="park"];
            node{0}["leisure"="dance"];
            node{0}["leisure"="fitness_centre"];
            way{0}["amenity"="cafe"];
            way{0}["amenity"="bar"];
            way{0}["amenity"="restaurant"];
            way{0}["amenity"="cinema"];
            way{0}["leisure"="park"];
            way{0}["leisure"="dance"];
            way{0}["leisure"="fitness_centre"];
            {1}
        );
        out center;
        """.format(
        house_area, q_employment
    )

    overpass_url = "http://overpass-api.de/api/interpreter"

    for try_no in range(3):
        response = requests.get(overpass_url, params={"data": q}, timeout=5 * 60)
        if try_no == 3:
            print_msg = "Too many Overpass requests in a short time."
            print_msg += " Please try again in a minute."
            print(print_msg)
            sys.exit()
        elif response.status_code == 200:
            break
        else:
            time.sleep(60)

    res = response.json()

    all_nodes = {
        "cafe": [],
        "bar": [],
        "restaurant": [],
        "cinema": [],
        "park": [],
        "dance": [],
        "fitness": [],
        "office": [],
        "university": [],
    }

    for element in res["elements"]:
        if element["type"] == "node":
            lon = element["lon"]
            lat = element["lat"]
        elif "center" in element:
            lon = element["center"]["lon"]
            lat = element["center"]["lat"]

        if "office" in element["tags"]:
            all_nodes["office"].append((lat, lon))

        if "amenity" in element["tags"]:
            if element["tags"]["amenity"] == "cafe":
                all_nodes["cafe"].append((lat, lon))
            if element["tags"]["amenity"] == "bar":
                all_nodes["bar"].append((lat, lon))
            if element["tags"]["amenity"] == "restaurant":
                all_nodes["restaurant"].append((lat, lon))
            if element["tags"]["amenity"] == "cinema":
                all_nodes["cinema"].append((lat, lon))
            if element["tags"]["amenity"] == "university":
                all_nodes["university"].append((lat, lon))
        elif "leisure" in element["tags"]:
            if element["tags"]["leisure"] == "park":
                all_nodes["park"].append((lat, lon))
            if element["tags"]["leisure"] == "dance":
                all_nodes["dance"].append((lat, lon))
            if element["tags"]["leisure"] == "fitness_centre":
                all_nodes["fitness"].append((lat, lon))

    return all_nodes


@pytest.fixture(scope="session")
def a_person(all_nodes):
    """
    Create a fixed Person instance
    """
    house_address = rndm_coords
    attributes = [0, 0, 7, 6, ["cafe", "bar", "park"]]

    return Person(house_address, attributes, all_nodes)


def test_set_travelling_status(a_person):
    """
    Test changing travelling status
    """

    a_person.set_travelling_status(5)
    assert a_person.travelling_status == 5


def test_set_active_status(a_person):
    """
    Test changing active status
    """

    a_person.set_active_status(2)
    assert a_person.active_status == 2


def test_update_preferred_exits_case_first_option(a_person):
    """
    Test changing preferred exits
    change first to second
    """

    a_person.update_preferred_exits("cafe")
    assert a_person.preferred_exits_today == ["bar", "cafe", "park"]


def test_update_preferred_exits_case_last_option(a_person):
    """
    Test changing preferred exits
    remove last
    """

    a_person.update_preferred_exits("park")
    assert (
        a_person.preferred_exits_today[:-1] == ["bar", "cafe"]
        and a_person.preferred_exits_today[-1] != "park"
    )


def test_update_preferred_exits_case_no_option(a_person):
    """
    Test changing preferred exits
    when selected exit not in preferred
    """

    exits_today = a_person.preferred_exits_today
    a_person.update_preferred_exits("not_an_option")
    assert a_person.preferred_exits_today == exits_today


def test_choose_preferred_exit_morning_home(a_person):
    """
    Test choosing preferred exit
    early in the morning
    """

    preferred_exit, location = a_person.choose_preferred_exit(0)
    assert preferred_exit == "home" and location == a_person.house_address


def test_choose_preferred_exit_night_home(a_person):
    """
    Test choosing preferred exit
    late in the night
    """

    preferred_exit, location = a_person.choose_preferred_exit(24 * 3600 - 1)
    assert preferred_exit == "home_night" and location == a_person.house_address


def test_choose_preferred_exit_random_exit(a_person):
    """
    Test choosing preferred exit
    random time
    """

    a_person.set_active_status(10)
    preferred_exit, location = a_person.choose_preferred_exit(12 * 3600)
    possible_exits = a_person.possible_exits + ["home", "home_night"]
    assert preferred_exit in possible_exits


def test_end_of_day_reset(a_person):
    """
    Test end of day reset
    of preferred exits
    """

    a_person.end_of_day_reset()
    assert a_person.preferred_exits == a_person.preferred_exits_today


def test_choose_action_day_home_action(a_person):
    """
    Test choosing action
    early at home
    """

    action, _, _, _ = a_person.choose_action(0, 0)
    assert action == "p"


def test_choose_action_day_home_location(a_person):
    """
    Test choosing action
    early at home
    """

    _, location, _, _ = a_person.choose_action(0, 0)
    assert location == a_person.house_address


def test_choose_action_day_home_exit(a_person):
    """
    Test choosing action
    early at home
    """

    _, _, _, exit_chosen = a_person.choose_action(0, 0)
    assert exit_chosen == "home_morning"


def test_choose_action_day_night_action(a_person):
    """
    Test choosing action
    late at home
    """

    action, _, _, _ = a_person.choose_action(23.5 * 3600, 2)
    assert action == "p"


def test_choose_action_day_night_location(a_person):
    """
    Test choosing action
    late at home
    """

    _, location, _, _ = a_person.choose_action(23.5 * 3600, 2)
    assert location == a_person.house_address


def test_choose_action_day_night_exit(a_person):
    """
    Test choosing action
    late at home
    """

    _, _, _, exit_chosen = a_person.choose_action(23.5 * 3600, 2)
    assert exit_chosen == "home_night"


def test_choose_action_simple_case_actions(a_person):
    """
    Test choosing action
    random time
    """

    action, _, _, _ = a_person.choose_action(15 * 3600, 2)
    assert action in ["p", "p_night", "fpf"]


def test_choose_action_simple_case_exit(a_person):
    """
    Test choosing action
    random time
    """

    _, _, _, exit_chosen = a_person.choose_action(15 * 3600, 2)
    assert exit_chosen in ["home_morning", "home_night"] + a_person.possible_exits


def test_choose_action_simple_case_times(a_person):
    """
    Test choosing action
    random time
    """

    _, _, times, _ = a_person.choose_action(15 * 3600, 2)
    assert times[1] >= times[0]


@pytest.fixture(scope="session")
def trajectory(a_person, all_nodes):
    """
    Generating fixed trajectory
    """

    house_address = a_person.house_address
    attributes = [0, 0, 7, 6, ["cafe", "bar", "park"]]
    switches = {}
    start_date = datetime.date(2021, 10, 1)
    end_date = datetime.date(2021, 10, 5)
    api_key = "5b3ce3597851110001cf6248551c505f7c61488a887356ff5ea924d5"
    traj, home_time_list, total_d_list = gen_all_traj(
        house_address,
        attributes,
        switches,
        all_nodes,
        start_date,
        end_date,
        api_key,
    )
    return traj, home_time_list, total_d_list


def test_gen_all_traj_len(trajectory):
    """
    Testing length
    """

    traj = trajectory[0]
    assert len(traj) == 4 * 24 * 3600


def test_gen_all_traj_time(trajectory):
    """
    Testing time is increasing
    """

    traj = trajectory[0]
    assert np.all(np.diff(traj[:, 0]) > 0)


def test_gen_all_traj_consistent_values(trajectory):
    """
    Testing consistent lattitude, longitude values
    """

    traj = trajectory[0]
    distances = []
    for i in range(len(traj) - 1):
        distances.append(
            great_circle_dist(traj[i, 1], traj[i, 2], traj[i + 1, 1], traj[i + 1, 2])
        )
    assert np.max(distances) <= 100


def test_gen_all_traj_time_at_home(trajectory):
    """
    Test home time in normal range
    """

    home_time_list = np.array(trajectory[1])
    assert np.all(home_time_list >= 0) and np.all(home_time_list <= 24 * 3600)


def test_gen_all_traj_dist_travelled(trajectory):
    """
    Test distance travelled in normal range
    """

    total_d_list = np.array(trajectory[2])
    assert np.all(total_d_list >= 0)


attributes = {
    "User 1": {
        "main_employment": "none",
        "vehicle": "car",
        "travelling_status": 10,
        "active_status": 8,
    },
    "User 5": {
        "main_employment": "work",
        "vehicle": "foot",
        "travelling_status": 9,
        "travelling_status-20": 1,
        "preferred_exits": ["cafe", "bar", "cinema"],
    },
}


def test_process_attributes_user_missing_args():
    """
    Test processing attributes
    with missing arguments
    """

    key = "User 1"
    user = 1
    attrs, _ = process_attributes(attributes, key, user)
    assert len(attrs) == 5


def test_process_attributes_arguments_correct():
    """
    Test processing attributes
    with missing arguments
    """

    key = "User 5"
    user = 5
    attrs, _ = process_attributes(attributes, key, user)
    assert attrs[0] == 0 and attrs[-1] == ["cafe", "bar", "cinema"]


def test_process_attributes_switch():
    """
    Test processing attributes
    with switch of behavior
    """

    key = "User 5"
    user = 5
    _, switches = process_attributes(attributes, key, user)
    assert (
        list(switches.keys())[0] == "travelling_status-20"
        and list(switches.values())[0] == 1
    )


@pytest.fixture(scope="session")
def final_df():
    """
    Creating final dataframe
    of trajectories
    """

    number_of_people = 1
    location = "US/Boston"
    start_date = [2021, 1, 1]
    end_date = [2021, 1, 8]
    cycle = 15
    percentage = 0.8
    api_key = "5b3ce3597851110001cf6248551c505f7c61488a887356ff5ea924d5"
    data_folder = tempfile.mkdtemp()

    sim_gps_data(
        number_of_people,
        location,
        start_date,
        end_date,
        cycle,
        percentage,
        api_key,
        data_folder,
    )

    data, _, _ = read_data("user_1", data_folder, "gps", "UTC", None, None)

    shutil.rmtree(data_folder)

    return data


def test_sim_gps_data_time(final_df):
    """
    Test simulating gps data
    increasing time
    """

    assert np.all(np.diff(final_df.iloc[:, 0]) > 0)


def test_sim_gps_data_data_points(final_df):
    """
    Test simulating gps data
    correct number of observations
    """

    assert final_df.shape[0] <= 7 * 24 * 3600 * 0.2


def test_sim_gps_data_data_consistent(final_df):
    """
    Test simulating gps data
    consistent lattitude, longitude
    """

    distances = []
    for i in range(len(final_df) - 1):
        distances.append(
            great_circle_dist(
                final_df.iloc[i, 2],
                final_df.iloc[i, 3],
                final_df.iloc[i + 1, 2],
                final_df.iloc[i + 1, 3],
            )
        )
    assert np.max(distances) <= 5000
