SYSTEM_PROMPT = """
You are a knowledgeable road cycling coach. You have been asked to help a cyclist who needes to decide on what to wear while training outside.
"""

USER_PROMPT = """
I have a training ride outside. It is {temperature} degrees Celsius, there is a {precipitation}% chance of rain, and the wind is blowing at {wind_speed} kph.
I have following clothes to choose from: {clothes}. What should I wear in order to stay comfortable during the ride?  
Only consider the weather conditions and the clothes I have mentioned. 
Output the clothes I should wear in the following format: "You should wear: [clothes]".
Do not provide any additional information. The output should only consist of one sentence.

"""
