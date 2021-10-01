"""
Tests for traj2stats summary statistics
in Jasmine
"""
import tempfile
import shutil
import json
from shapely.geometry import Point
import pytest
import pandas as pd
import numpy as np
from forest.bonsai.simulate_gps_data import sim_gps_data
from forest.jasmine.traj2stats import (
    transform_point_to_circle,
    gps_stats_main,
)
from forest.jasmine.data2mobmat import great_circle_dist

coords1 = [51.457183, -2.597960]
coords2 = [51.457267, -2.598045]
coords3 = [51.457488, -2.598425]


def test_transform_point_to_circle_simple_case():
    """
    Testing creating a circle from
    a center point in coordinates
    """
    circle1 = transform_point_to_circle(coords1, 15)
    point2 = Point(coords2)
    assert circle1.contains(point2)


def test_transform_point_to_circle_zero_radius():
    """
    Testing creating a circle from
    a center point in coordinates
    with zero radius
    """
    circle1 = transform_point_to_circle(coords1, 0)
    assert len(circle1.exterior.coords) == 0


def test_transform_point_to_circle_radius():
    """
    Testing creating a circle from
    a center point in coordinates
    and checking radius is approximately
    correct
    """

    circle1 = transform_point_to_circle(coords1, 5)
    point_in_edge = [circle1.exterior.coords.xy[0][2], circle1.exterior.coords.xy[1][2]]

    distance = great_circle_dist(
        coords1[0], coords1[1], point_in_edge[0], point_in_edge[1]
    )
    assert distance >= 4 and distance <= 5


@pytest.fixture(scope="session")
def simulated_trajectories():
    """
    Creating simulated trajectories
    """

    number_of_people = 1
    location = "GB/Bristol"
    start_date = [2021, 1, 1]
    end_date = [2021, 1, 8]
    cycle = 15
    percentage = 0.8
    api_key = "5b3ce3597851110001cf6248551c505f7c61488a887356ff5ea924d5"
    data_folder = tempfile.mkdtemp()
    attributes_folder = tempfile.mkdtemp()
    attributes = {"User 1": {"active_status": 8}}

    with open(attributes_folder + "/attributes.json", "w") as attr_dir:
        json.dump(attributes, attr_dir)

    sim_gps_data(
        number_of_people,
        location,
        start_date,
        end_date,
        cycle,
        percentage,
        api_key,
        data_folder,
        attributes_folder + "/attributes.json",
    )

    shutil.rmtree(attributes_folder)

    output_folder1 = tempfile.mkdtemp()
    output_folder2 = tempfile.mkdtemp()
    gps_stats_main(
        data_folder,
        output_folder1,
        "Etc/GMT+1",
        "both",
        True,
        places_of_interest=["cafe", "bar", "park"],
        save_log=True,
        threshold=60,
    )

    gps_stats_main(
        data_folder,
        output_folder2,
        "Etc/GMT+1",
        "daily",
        True,
        places_of_interest=["restaurant", "cinema", "park"],
        split_day_night=True,
    )

    shutil.rmtree(data_folder)

    return output_folder1, output_folder2


@pytest.fixture(scope="session")
def daily_stats(simulated_trajectories):
    """
    Loading daily summary stats
    """
    res = pd.read_csv(simulated_trajectories[0] + "/daily/user_1.csv")
    return res


@pytest.fixture(scope="session")
def hourly_stats(simulated_trajectories):
    """
    Loading hourly summary stats
    """
    res = pd.read_csv(simulated_trajectories[0] + "/hourly/user_1.csv")
    return res


@pytest.fixture(scope="session")
def daily_log(simulated_trajectories):
    """
    Loading daily logs json file
    """
    with open(simulated_trajectories[0] + "/daily/locations_logs.json") as logs:
        res = json.load(logs)
    return res


@pytest.fixture(scope="session")
def datetime_nighttime_stats(simulated_trajectories):
    """
    Loading daily summary stats
    with datetime nighttime patterns
    """
    res = pd.read_csv(simulated_trajectories[1] + "/user_1.csv")
    return res


def test_summary_stats_daily_shape(daily_stats):
    """
    Testing shape of daily summary stats
    """
    assert daily_stats.shape == (8, 28)


def test_summary_stats_daily_missing(daily_stats):
    """
    Testing missing values of daily summary stats
    """
    assert sum(daily_stats.isna().sum()) == 0


def test_summary_stats_places_of_interest(daily_stats):
    """
    Testing amount of time spent in places
    of interest less than pause time
    """
    time_in_places_of_interest = (
        daily_stats["cafe"]
        + daily_stats["bar"]
        + daily_stats["park"]
        + daily_stats["other"]
    )
    assert np.all(time_in_places_of_interest <= daily_stats["total_pause_time"])


def test_summary_stats_places_of_interest_adjusted(daily_stats):
    """
    Testing total adjusted time spent at places of interest
    is equal with total non adjusted
    """
    time_in_places_of_interest = (
        daily_stats["cafe"] + daily_stats["bar"] + daily_stats["park"]
    )

    time_in_places_of_interest_adjusted = (
        daily_stats["cafe_adjusted"]
        + daily_stats["bar_adjusted"]
        + daily_stats["park_adjusted"]
    )
    assert np.all(time_in_places_of_interest == time_in_places_of_interest_adjusted)


def test_summary_stats_log_format(daily_stats, daily_log):
    """
    Testing json logs contain all
    dates from summary stats
    """
    dates_stats = (
        daily_stats["day"].astype(int).astype(str)
        + "/"
        + daily_stats["month"].astype(int).astype(str)
        + "/"
        + daily_stats["year"].astype(int).astype(str)
    )
    dates_log = np.array(list(daily_log.keys()))
    assert np.all(dates_stats == dates_log)


def test_summary_stats_obs_day_night(daily_stats):
    """
    Testing total observation time is same
    as day observation plus night observation times
    """
    total_obs = daily_stats["obs_day"] + daily_stats["obs_night"]
    assert np.all(round(total_obs, 4) == round(daily_stats["obs_duration"], 4))


def test_summary_stats_hourly_shape(hourly_stats):
    """
    Testing shape of hourly summary stats
    """
    assert hourly_stats.shape == (165, 23)


def test_summary_stats_hourly_missing(hourly_stats):
    """
    Testing missing values of hourly summary stats
    """
    assert sum(hourly_stats.isna().sum()) == 0


def test_summary_stats_hourly_time(hourly_stats):
    """
    Testing all times included in hourly summary stats
    """
    for i in range(len(hourly_stats["hour"]) - 1):
        if hourly_stats["hour"][i] == 23 and hourly_stats["hour"][i + 1] == 0:
            pass
        elif (hourly_stats["hour"][i] + 1) == hourly_stats["hour"][i + 1]:
            pass
        else:
            assert False
    assert True


def test_summary_stats_datetime_nighttime_shape(datetime_nighttime_stats):
    """
    Testing shape of datetime nighttime summary stats
    """
    assert datetime_nighttime_stats.shape == (8, 50)


def test_summary_stats_datetime_nighttime_validity(datetime_nighttime_stats):
    """
    Testing datetime nighttime patterns in places of interest
    are less than their respective pause times
    """

    time_in_places_of_interest_datetime = (
        datetime_nighttime_stats["restaurant_datetime"]
        + datetime_nighttime_stats["cinema_datetime"]
        + datetime_nighttime_stats["park_datetime"]
        + datetime_nighttime_stats["other_datetime"]
    )
    time_in_places_of_interest_nighttime = (
        datetime_nighttime_stats["restaurant_nighttime"]
        + datetime_nighttime_stats["cinema_nighttime"]
        + datetime_nighttime_stats["park_nighttime"]
        + datetime_nighttime_stats["other_nighttime"]
    )
    boolean1 = round(time_in_places_of_interest_datetime, 4) <= round(
        datetime_nighttime_stats["total_pause_time_datetime"], 4
    )
    boolean2 = round(time_in_places_of_interest_nighttime, 4) <= round(
        datetime_nighttime_stats["total_pause_time_nighttime"], 4
    )
    assert np.all(boolean1) and np.all(boolean2)
