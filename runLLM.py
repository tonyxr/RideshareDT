#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Xiaoru Shi
"""

from LLMAgent import LLMCustomerAgent

# Create agent
class Core: 
    def __init__(self, machineID):
        self.agent = LLMCustomerAgent()
        self.machineID = machineID
        
    def runInstance(self):

        # Define a sample synthetic customer
        result = self.agent.evaluate_customer(
            age=30,
            income="100k-200k",
            city="Chicago",
            platform="Lyft",
            household_size=3,
            marital_status="Married",
            employment_status="Employed",
            travel_distance=4.2,
            ride_price=12.50
        )

        # Output
        print("Prompt:\n", result["Prompt"])
        print("\nResponse:\n", result["Response"])
        print("\nAccepted Ride?:", result["Accepted"])

if __name__ == '__main__':
    print("Enter the number of replications")
    replications = int(input())
    print("Enter machine index")
    machine = input()
    customerLLM = Core(machine)
    for i in range(0, replications):
        customerLLM.runInstance()