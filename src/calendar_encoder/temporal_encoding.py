import math
import ephem
import numpy as np
import pandas as pd
from scipy.fftpack import fft
from sklearn.preprocessing import MinMaxScaler


class Time2Vec:

    def __init__(self, col_time, col_target):
        self.min_year = 1900
        self.max_year = 2100
        self.min_month = 1
        self.max_month = 12
        self.min_day = 1
        self.max_day = 31
        self.min_week = 1
        self.max_week = 52
        self.min_day_of_week = 0
        self.max_day_of_week = 6
        self.min_minute = 0
        self.max_minute = 59
        self.min_second = 0
        self.max_second = 59
        self.min_hour = 0
        self.max_hour = 23
        self.scaler = MinMaxScaler()
        self.col_time = col_time
        self.col_target = col_target

    def get_part_of_day(self, hour):
        if 6 <= hour < 12:
            return 0
        elif 12 <= hour < 18:
            return 1
        elif 18 <= hour < 22:
            return 2
        else:
            return 3

    def check_different_years(self):
        return self.min_year != self.max_year

    def get_season(self, month):
        if month in [12, 1, 2]:
            return 0  # Зима
        elif month in [3, 4, 5]:
            return 1  # Весна
        elif month in [6, 7, 8]:
            return 2  # Лето
        else:
            return 3

    def meta_date(self, df):
        df_with_meta = df.copy()
        df_with_meta[self.col_time] = pd.to_datetime(df_with_meta[self.col_time])
        df_with_meta.set_index(self.col_time, inplace=True)
        df_with_meta[self.col_time] = df[self.col_time]
        df_with_meta['year'] = df_with_meta.index.year
        df_with_meta['month'] = df_with_meta.index.month
        df_with_meta['day'] = df_with_meta.index.day
        df_with_meta['week'] = df_with_meta.index.isocalendar().week
        df_with_meta['day_of_week'] = df_with_meta.index.dayofweek
        df_with_meta['hour'] = df_with_meta.index.hour
        df_with_meta['minute'] = df_with_meta.index.minute
        df_with_meta['second'] = df_with_meta.index.second
        df_with_meta['hour_sin'] = np.sin(2 * np.pi * df_with_meta['hour'] / 24)
        df_with_meta['hour_cos'] = np.cos(2 * np.pi * df_with_meta['hour'] / 24)
        df_with_meta['day_of_week_sin'] = np.sin(2 * np.pi * df_with_meta['day_of_week'] / 7)
        df_with_meta['day_of_week_cos'] = np.cos(2 * np.pi * df_with_meta['day_of_week'] / 7)
        df_with_meta['week_sin'] = np.sin(2 * np.pi * df_with_meta['week'] / 52)
        df_with_meta['week_cos'] = np.cos(2 * np.pi * df_with_meta['week'] / 52)
        df_with_meta['month_sin'] = np.sin(2 * np.pi * df_with_meta['month'] / 12)
        df_with_meta['month_cos'] = np.cos(2 * np.pi * df_with_meta['month'] / 12)
        df_with_meta['part_of_day'] = df_with_meta['hour'].apply(self.get_part_of_day)
        df_with_meta['is_night'] = df_with_meta['hour'].apply(lambda x: 1 if x >= 22 or x < 6 else 0)
        df_with_meta['is_weekend'] = df_with_meta['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
        df_with_meta['day_of_year'] = df_with_meta.index.dayofyear

        df_with_meta['is_working_hours'] = df_with_meta.apply(lambda row: 1 if 9 <= row['hour'] < 18 and row['is_weekend'] == 0 else 0, axis=1)
        df_with_meta['season'] = df_with_meta['month'].apply(self.get_season)
        df_with_meta['season_sin'] = np.sin(2 * np.pi * df_with_meta['season'] / 4)
        df_with_meta['season_cos'] = np.cos(2 * np.pi * df_with_meta['season'] / 4)
        df_with_meta['quarter'] = df_with_meta.index.quarter
        df_with_meta['quarter_sin'] = np.sin(2 * np.pi * df_with_meta['quarter'] / 4)
        df_with_meta['quarter_cos'] = np.cos(2 * np.pi * df_with_meta['quarter'] / 4)
        df_with_meta['moon_phase'] = df_with_meta.index.to_series().apply(lambda x: ephem.Moon(x).phase / 29.53)

        # df_with_meta['time_trend'] = (df_with_meta.index - df_with_meta.index.min()).total_seconds()
        df_with_meta['time_trend'] = (
                (df_with_meta.index - df_with_meta.index.min()).total_seconds()
                / (df_with_meta.index.max() - df_with_meta.index.min()).total_seconds()
        )

        return df_with_meta

    def inverse_normalize_column(self, column, min_val, max_val):
        return column * (max_val - min_val) + min_val


    def encoder(self, df):
        all_col = df.columns
        col_vec = [
            self.col_time,
            self.col_target,
            "year",
            "month",
            "day",
            "week",
            "day_of_week",
            "hour",
            "minute",
            "second",
            "hour_sin",
            "hour_cos",
            "day_of_week_sin",
            "day_of_week_cos",
            "week_sin",
            "week_cos",
            "month_sin",
            "month_cos",
            "part_of_day",
            "is_night",
            "is_weekend",
            "day_of_year",
            "is_working_hours",
            "season",
            "season_sin",
            "season_cos",
            "quarter",
            "quarter_sin",
            "quarter_cos",
            "moon_phase",
            "time_trend",
        ]

        diff_cols = list(all_col.difference(col_vec))

        df[self.col_target] = df[self.col_target].astype(float)
        min_val = df[self.col_target].min() * 1.2
        max_val = df[self.col_target].max() * 1.2

        df_with_meta = self.meta_date(df)
        normalized_dates = []

        for index, date in df_with_meta.iterrows():
            time = index
            diff_col_values = date[diff_cols].values.tolist()

            year_norm = (date['year'] - self.min_year) / (self.max_year - self.min_year) if self.check_different_years() else 1
            month_norm = (date['month'] - self.min_month) / (self.max_month - self.min_month)
            day_norm = (date['day'] - self.min_day) / (self.max_day - self.min_day)
            week_norm = (date['week'] - self.min_week) / (self.max_week - self.min_week)
            day_of_week_norm = (date['day_of_week'] - self.min_day_of_week) / (self.max_day_of_week - self.min_day_of_week)
            hour_norm = (date['hour'] - self.min_hour) / (self.max_hour - self.min_hour)
            minute_norm = (date['minute'] - self.min_minute) / (self.max_minute - self.min_minute)
            second_norm = (date['second'] - self.min_second) / (self.max_second - self.min_second)
            part_of_day_norm = (date['part_of_day'] - 0) / 3
            is_night_norm = date['is_night']
            is_weekend_norm = date['is_weekend']
            day_of_year_norm = (date['day_of_year'] - 1) / 365

            is_working_hours = date["is_working_hours"]
            season = date["season"]
            season_sin = date["season_sin"]
            season_cos = date["season_cos"]
            quarter = date["quarter"]
            quarter_sin = date["quarter_sin"]
            quarter_cos = date["quarter_cos"]
            moon_phase = date["moon_phase"]
            time_trend = date["time_trend"]

            normalized_date = [
                                  time, date[self.col_target], year_norm, month_norm, day_norm, week_norm, day_of_week_norm,
                                  hour_norm, minute_norm, second_norm,
                                  date['hour_sin'], date['hour_cos'], date['day_of_week_sin'], date['day_of_week_cos'],
                                  date['week_sin'], date['week_cos'], date['month_sin'], date['month_cos'],
                                  part_of_day_norm, is_night_norm, is_weekend_norm, day_of_year_norm, is_working_hours,
                                  season, season_sin, season_cos, quarter, quarter_sin, quarter_cos, moon_phase, time_trend
                              ] + diff_col_values
            normalized_dates.append(normalized_date)

        normalized_df = pd.DataFrame(normalized_dates, columns=col_vec + diff_cols)
        # normalized_df[self.col_target] = self.normalize_column(normalized_df[self.col_target], min_val, max_val)
        cols = [c for c in normalized_df.columns if c != self.col_time]
        normalized_df[cols] = normalized_df[cols].apply(pd.to_numeric, errors="coerce").astype("float64")

        return normalized_df, min_val, max_val

    def decoder(self, df, min_val, max_val):
        df = df.sort_values(by=['year', 'month', 'day', 'hour', 'minute'], ascending=True)

        denormalized_dates = []
        for index, date in df.iterrows():
            year_denorm = date['year'] * (self.max_year - self.min_year) + self.min_year
            month_denorm = date['month'] * (self.max_month - self.min_month) + self.min_month
            day_denorm = date['day'] * (self.max_day - self.min_day) + self.min_day
            week_denorm = date['week'] * (self.max_week - self.min_week) + self.min_week
            day_of_week_denorm = date['day_of_week'] * (self.max_day_of_week - self.min_day_of_week) + self.min_day_of_week
            hour_denorm = date['hour'] * (self.max_hour - self.min_hour) + self.min_hour
            minute_denorm = date['minute'] * (self.max_minute - self.min_minute) + self.min_minute
            second_denorm = date['second'] * (self.max_second - self.min_second) + self.min_second

            denormalized_date = [
                date[self.col_target], year_denorm, month_denorm, day_denorm, week_denorm,
                day_of_week_denorm, hour_denorm, minute_denorm, second_denorm
            ]
            denormalized_dates.append(denormalized_date)

        for i in range(len(denormalized_dates)):
            denormalized_dates[i][0] = self.inverse_normalize_column(denormalized_dates[i][0], min_val, max_val)

        denormalized_df = pd.DataFrame(denormalized_dates, columns=[
            self.col_target, 'year', 'month', 'day', 'week', 'day_of_week',
            'hour', 'minute', 'second'
        ])

        denormalized_df['hour'] = denormalized_df['hour'].apply(lambda x: math.ceil(x))
        denormalized_df['minute'] = denormalized_df['minute'].apply(lambda x: math.ceil(x))
        denormalized_df['second'] = denormalized_df['second'].apply(lambda x: math.ceil(x))
        denormalized_df['month'] = denormalized_df['month'].apply(lambda x: math.ceil(x))
        denormalized_df['day'] = denormalized_df['day'].apply(lambda x: math.ceil(x))
        denormalized_df['year'] = denormalized_df['year'].apply(lambda x: math.ceil(x))

        denormalized_df[self.col_time] = pd.to_datetime({
            'year': denormalized_df['year'],
            'month': denormalized_df['month'],
            'day': denormalized_df['day'],
            'hour': denormalized_df['hour'],
            'minute': denormalized_df['minute'],
            'second': denormalized_df['second']
        })

        return denormalized_df


    def easy_decoder(self, df, min_val, max_val):
        df = df[[self.col_target]]
        df[self.col_target] = self.inverse_normalize_column(df[self.col_target], min_val, max_val)
        return df

