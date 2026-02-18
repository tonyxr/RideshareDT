#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Xiaoru Shi


"""

import pandas as pd
import numpy as np

# Load the CSV of Seattle Uber customers
df = pd.read_csv("seattle_uber_customers.csv")

class ComputeUtility:
    def __init__(self, 
                 base_fare: float, 
                 per_mile: float, 
                 per_min: float, 
                 booking_fee: float, 
                 avg_speed_mph: float, 
                 priceElasticity: float,
                 taste: float,
                 random_seed: int | None = 42,
                 ):
        
        self.base_fare = base_fare
        self.per_mile = per_mile
        self.per_min = per_min
        self.booking_fee = booking_fee
        self.avgSpeed = max(15.0, avg_speed_mph)
        self.priceElasticity = priceElasticity
        self.taste = float(taste)
        self.rng = np.random.default_rng(random_seed)
        
        self.incomePriceMultiplier = {
            '<50k': 1.35,
            '50k-100k': 1.10,
            '100k-200k': 0.85,
            '200k+': 0.70
        }
        
        self.beta_income_level = {
            '<50k': -0.10,
            '50k-100k': 0.00,
            '200k+': 0.15
        }
        
        self.beta_marital = {
            'Single': 0.10, 
            'Married': 0.12, 
            'Divorced': 0.05, 
            'Widowed': 0.04
        }
        
        self.beta_employment = {
            'Employed': 0.20,
            'Unemployed': -0.05,
            'Student': 0.12,
            'Retired': 0.02
        }
        
        self.beta_household = 0.08
        self.beta_log_distance = -0.25
        self.beta_age_linear = -0.003
        self.beta_age_quadr = -0.00008

    def checkPrice(self, priceAction: pd.DataFrame) -> np.ndarray:
        if 'Price' in df.columns:
            price = np.asarray(df['Price'], dtype = float)
        else:
            miles = np.asarray(df['TravelDistance'], dtype = float)
            minutes = 60.0 * miles / self.avgSpeed
            price = (self.base_fare + self.booking_fee + self.per_mile * miles + self.per_min * minutes)
    
   
    """
    def __init__(self, price=15.0):
        self.price = price

    def compute(self, df):
        def utility_function(row):
            beta_age = -0.005 * row['Age']  # softer age penalty
            beta_distance = -0.15 * row['TravelDistance']  # milder distance cost

            beta_income = {
                '<50k': -0.3,
                '50k-100k': -0.1,
                '100k-200k': 0.1,
                '200k+': 0.25
            }.get(row['IncomeBracket'], 0.0)
            
            beta_household = 0.15 * row['HouseholdSize']  # stronger weight
            beta_status = {
                'Single': 0.2,
                'Married': 0.15,
                'Divorced': 0.1,
                'Widowed': 0.1
            }.get(row['MaritalStatus'], 0.1)

            beta_employment = {
                'Employed': 0.4,
                'Unemployed': 0.05,
                'Student': 0.3,
                'Retired': 0.1
            }.get(row['EmploymentStatus'], 0.1)

            epsilon = np.random.normal(0, 1)  # random taste variation

            price_penalty = 0.7 * self.price  # reduce sensitivity to price

            utility = (
                beta_age +
                beta_income +
                beta_household +
                beta_status +
                beta_employment -
                beta_distance -
                price_penalty +
                epsilon
            )
            return utility

        df['Utility'] = df.apply(utility_function, axis=1)
        df['PurchaseDecision'] = df['Utility'] > 0
        return df
    """