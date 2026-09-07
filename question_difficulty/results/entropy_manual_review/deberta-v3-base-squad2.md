# Attention-Entropy Difficulty Signal — Raw Cell Outputs

**Model:** `deepset/deberta-v3-base-squad2`
**Layer:** 11
**Source notebook:** `question_difficulty/notebooks/entropy_manual_review.ipynb`
**Date:** 2026-09-04

Raw output only, copied verbatim from each cell. No interpretation.

§3 (main-corpus extraction / length-confound Spearman check) has no cached output in the notebook
as of this export.

## §1 output

```
Loaded 52 CACHED samples from entropy_review_samples.json (delete this file to re-sample)
14 unique passages, 52 total questions (3.7 questions/passage avg)
  [RACE-middle] This is an orange   _  .
  [RACE-middle] The pen is   _  .
  [RACE-middle] _   pencils are in the pencil case.
```

## §5 output — literal vs. inferential

```
====================================================================================================
PASSAGE: Lily always carries a spare phone charger in her bag. Her phone battery tends to die quickly during long days at work.

  LITERAL      Q: What does Lily always carry in her bag?
               gold='a spare phone charger'
               tok_entropy_norm=0.848  sent_entropy_norm=0.585  question_answer_overlap_coef=0.000
               predicted_answer='a spare phone charger'  confidence=0.714  f1=1.000  CORRECT

  INFERENTIAL  Q: Why won't Lily be stuck with a dead phone during a long day at work?
               gold='a spare phone charger'
               tok_entropy_norm=0.949  sent_entropy_norm=0.999  question_answer_overlap_coef=0.333
               predicted_answer='spare phone charger'  confidence=0.000  f1=0.857  CORRECT

  deltas vs LITERAL:
    INFERENTIAL  tok_entropy_norm=+0.100  sent_entropy_norm=+0.414  overlap=+0.333  confidence=-0.714  f1=-0.143

====================================================================================================
PASSAGE: Every school day, Maria wakes up early. She needs to go to school at 9AM, so she prepares her books and uniform the night before. Her mother drops her off on the way to work.

  LITERAL      Q: What does she need to do at 9AM?
               gold='go to school'
               tok_entropy_norm=0.887  sent_entropy_norm=0.814  question_answer_overlap_coef=0.000
               predicted_answer='go to school'  confidence=0.968  f1=1.000  CORRECT

  INFERENTIAL  Q: Why can't she wake up at 10AM?
               gold='she needs to go to school at 9AM'
               tok_entropy_norm=0.922  sent_entropy_norm=0.935  question_answer_overlap_coef=0.400
               predicted_answer='She needs to go to school at 9AM'  confidence=0.000  f1=0.875  CORRECT

  deltas vs LITERAL:
    INFERENTIAL  tok_entropy_norm=+0.034  sent_entropy_norm=+0.121  overlap=+0.400  confidence=-0.968  f1=-0.125

====================================================================================================
PASSAGE: Every winter, the small town of Millbrook loses power for a few hours during storms. Last year, the local government installed a backup generator at the community center to keep the building running during outages. The generator can supply electricity for up to 12 hours before needing more fuel. Residents can charge their phones and stay warm there when their home power goes out. Some elderly residents rely on the center to keep medical equipment running. The town is now considering installing solar panels to reduce fuel costs.

  LITERAL      Q: What did the local government install at the community center?
               gold='a backup generator'
               tok_entropy_norm=0.843  sent_entropy_norm=0.741  question_answer_overlap_coef=0.000
               predicted_answer='a backup generator'  confidence=0.958  f1=1.000  CORRECT

  INFERENTIAL  Q: Why can Millbrook residents stay warm and charge their phones at the community center during a power outage?
               gold='the generator can supply electricity'
               tok_entropy_norm=0.877  sent_entropy_norm=0.921  question_answer_overlap_coef=0.000
               predicted_answer='backup generator at the community center to keep the building running during outages. The generator can supply electricity for up to 12 hours before needing more fuel.'  confidence=0.011  f1=0.312  WRONG

  deltas vs LITERAL:
    INFERENTIAL  tok_entropy_norm=+0.034  sent_entropy_norm=+0.179  overlap=+0.000  confidence=-0.948  f1=-0.688

====================================================================================================
PASSAGE: In 2015, a small biotech company in Boston began testing a new type of bandage designed for chronic wounds. Unlike traditional bandages, this one contained a thin layer of sensors that measured moisture, temperature, and bacterial growth in real time. The data was sent wirelessly to a smartphone app, allowing doctors to monitor a patient's healing progress without requiring a clinic visit. Early trials showed that patients using the smart bandage healed nearly 20 percent faster than those using standard dressings. Doctors were able to adjust treatment plans within a day of an infection appearing, rather than waiting for the patient's next scheduled appointment. The bandage costs significantly more than a regular one, which has slowed its adoption in hospitals with tight budgets. Even so, several insurance companies have begun covering the cost for patients with diabetes, who are especially prone to slow-healing wounds. The company hopes to expand production and lower the price within the next two years.

  LITERAL      Q: What three things did the sensors in the bandage measure?
               gold='moisture, temperature, and bacterial growth'
               tok_entropy_norm=0.868  sent_entropy_norm=0.828  question_answer_overlap_coef=0.000
               predicted_answer='moisture, temperature, and bacterial growth'  confidence=0.997  f1=1.000  CORRECT

  INFERENTIAL  Q: Why were doctors able to adjust treatment plans within a day of an infection appearing?
               gold='sensors measured moisture, temperature, and bacterial growth in real time'
               tok_entropy_norm=0.888  sent_entropy_norm=0.844  question_answer_overlap_coef=0.000
               predicted_answer="rather than waiting for the patient's next scheduled appointment"  confidence=0.919  f1=0.000  WRONG

  deltas vs LITERAL:
    INFERENTIAL  tok_entropy_norm=+0.020  sent_entropy_norm=+0.016  overlap=+0.000  confidence=-0.078  f1=-1.000

====================================================================================================
PASSAGE: When Elena took over as principal of Riverside Elementary three years ago, nearly a third of the school's students were missing more than two weeks of class every year. After interviewing families, she discovered that many parents worked early shifts and had no way to get their children to school on time, since the nearest bus stop was almost two miles away. Elena partnered with a local transit company to add a new bus route that stopped directly in front of the school at 7:15 each morning. She also arranged for breakfast to be served starting at 7:00, so students arriving early would not have to wait outside in the cold. Within the first semester, chronic absenteeism dropped from 32 percent to 19 percent. Teachers reported that students who used to miss the first class of the day were now consistently present, which allowed lessons to build on each other more smoothly. The district was impressed enough that it approved funding to extend the new bus route to two neighboring schools the following year. Some parents initially worried that the earlier start time would be difficult for younger children, but a survey conducted at the end of the year found that most families adjusted within the first month. Elena says the biggest lesson from the program was that a single logistical barrier, transportation, was quietly undermining years of curriculum improvements the school had already made.

  LITERAL      Q: What time does the new bus route stop in front of the school?
               gold='7:15'
               tok_entropy_norm=0.844  sent_entropy_norm=0.791  question_answer_overlap_coef=0.000
               predicted_answer='7:15'  confidence=0.999  f1=1.000  CORRECT

  INFERENTIAL  Q: Why did chronic absenteeism drop after Elena introduced the new bus route?
               gold='the new bus route gave students a way to get to school on time'
               tok_entropy_norm=0.876  sent_entropy_norm=0.978  question_answer_overlap_coef=0.333
               predicted_answer='students who used to miss the first class of the day were now consistently present'  confidence=0.199  f1=0.207  WRONG

  deltas vs LITERAL:
    INFERENTIAL  tok_entropy_norm=+0.032  sent_entropy_norm=+0.187  overlap=+0.333  confidence=-0.800  f1=-0.793
```

## §6 output — negation

```
====================================================================================================
PASSAGE: The new employee handbook covers vacation policy, sick leave, and health insurance. It does not include any information about retirement benefits, since those are managed by a separate HR portal.

  POSITIVE     Q: What three topics does the employee handbook cover?
               gold='vacation policy, sick leave, and health insurance'
               tok_entropy_norm=0.832  sent_entropy_norm=0.683  question_answer_overlap_coef=0.000
               predicted_answer='vacation policy, sick leave, and health insurance'  confidence=0.999  f1=1.000  CORRECT

  NEGATION     Q: What topic does the employee handbook NOT include?
               gold='retirement benefits'
               tok_entropy_norm=0.847  sent_entropy_norm=0.823  question_answer_overlap_coef=0.000
               predicted_answer='retirement benefits'  confidence=0.999  f1=1.000  CORRECT

  deltas vs POSITIVE:
    NEGATION     tok_entropy_norm=+0.015  sent_entropy_norm=+0.140  overlap=+0.000  confidence=+0.000  f1=+0.000

====================================================================================================
PASSAGE: The gym's monthly membership includes access to the weight room and cardio machines, but it does not include personal training sessions.

  POSITIVE     Q: What does the gym membership include?
               gold='the weight room and cardio machines'
               tok_entropy_norm=0.939  sent_entropy_norm=0.000  question_answer_overlap_coef=0.000
               predicted_answer='access to the weight room and cardio machines'  confidence=1.000  f1=0.857  CORRECT

  NEGATION     Q: What does the gym membership NOT include?
               gold='personal training sessions'
               tok_entropy_norm=0.870  sent_entropy_norm=0.000  question_answer_overlap_coef=0.000
               predicted_answer='personal training sessions'  confidence=0.995  f1=1.000  CORRECT

  deltas vs POSITIVE:
    NEGATION     tok_entropy_norm=-0.069  sent_entropy_norm=+0.000  overlap=+0.000  confidence=-0.004  f1=+0.143

====================================================================================================
PASSAGE: The cafe's lunch menu offers soup, salad, and a daily sandwich special. It does not offer any hot entrees until dinner service begins at 5PM.

  POSITIVE     Q: What three items does the cafe's lunch menu offer?
               gold='soup, salad, and a daily sandwich special'
               tok_entropy_norm=0.832  sent_entropy_norm=0.504  question_answer_overlap_coef=0.000
               predicted_answer='soup, salad, and a daily sandwich special'  confidence=0.999  f1=1.000  CORRECT

  NEGATION     Q: What does the cafe's lunch menu NOT offer?
               gold='hot entrees'
               tok_entropy_norm=0.912  sent_entropy_norm=0.981  question_answer_overlap_coef=0.000
               predicted_answer='any hot entrees'  confidence=0.016  f1=0.800  CORRECT

  deltas vs POSITIVE:
    NEGATION     tok_entropy_norm=+0.081  sent_entropy_norm=+0.477  overlap=+0.000  confidence=-0.982  f1=-0.200

====================================================================================================
PASSAGE: The basic travel insurance plan covers trip cancellation, lost luggage, and emergency medical expenses abroad. It does not cover pre-existing medical conditions unless the traveler purchases an additional waiver within 14 days of booking. Many customers are surprised to learn this exclusion applies even to well-controlled conditions like diabetes or asthma. Travelers with chronic illnesses are strongly advised to read the fine print before relying on the basic plan alone.

  POSITIVE     Q: What three things does the basic travel insurance plan cover?
               gold='trip cancellation, lost luggage, and emergency medical expenses abroad'
               tok_entropy_norm=0.774  sent_entropy_norm=0.509  question_answer_overlap_coef=0.000
               predicted_answer='trip cancellation, lost luggage, and emergency medical expenses'  confidence=0.829  f1=0.941  CORRECT

  NEGATION     Q: What does the basic travel insurance plan NOT cover unless a waiver is purchased?
               gold='pre-existing medical conditions'
               tok_entropy_norm=0.826  sent_entropy_norm=0.710  question_answer_overlap_coef=0.000
               predicted_answer='pre-existing medical conditions'  confidence=0.999  f1=1.000  CORRECT

  deltas vs POSITIVE:
    NEGATION     tok_entropy_norm=+0.052  sent_entropy_norm=+0.201  overlap=+0.000  confidence=+0.170  f1=+0.059

====================================================================================================
PASSAGE: Starting next month, the city's curbside recycling program will accept paper, cardboard, glass bottles, and most plastics labeled 1 through 5. City officials spent the past year studying which materials could realistically be processed at the local sorting facility without expensive upgrades. The program will not accept plastic bags, styrofoam, or any electronics, since the facility lacks the equipment to safely process them and past attempts led to jammed machinery. Residents who want to dispose of electronics can instead drop them off at the county hazardous waste center, which accepts them free of charge on the first Saturday of every month. City officials estimate that clearly excluding these problematic materials from curbside pickup will reduce contamination rates at the sorting facility by nearly 40 percent, based on data from similar programs in neighboring cities.

  POSITIVE     Q: What four types of materials will the curbside recycling program accept?
               gold='paper, cardboard, glass bottles, and most plastics labeled 1 through 5'
               tok_entropy_norm=0.829  sent_entropy_norm=0.673  question_answer_overlap_coef=0.000
               predicted_answer='paper, cardboard, glass bottles, and most plastics labeled 1 through 5'  confidence=0.910  f1=1.000  CORRECT

  NEGATION     Q: What three things will the curbside recycling program NOT accept?
               gold='plastic bags, styrofoam, or any electronics'
               tok_entropy_norm=0.872  sent_entropy_norm=0.844  question_answer_overlap_coef=0.000
               predicted_answer='plastic bags, styrofoam, or any electronics'  confidence=0.960  f1=1.000  CORRECT

  deltas vs POSITIVE:
    NEGATION     tok_entropy_norm=+0.044  sent_entropy_norm=+0.172  overlap=+0.000  confidence=+0.050  f1=+0.000
```

## §7 output — comparison and superlative

```
====================================================================================================
PASSAGE: The Riverside factory produced 1,200 units in January and 1,850 units in March. The Lakeside factory produced 1,500 units in January and 1,600 units in March. The Hillcrest factory produced 1,300 units in January and 1,420 units in March.

  LITERAL      Q: How many units did the Riverside factory produce in March?
               gold='1,850'
               tok_entropy_norm=0.749  sent_entropy_norm=0.468  question_answer_overlap_coef=0.000
               predicted_answer='1,850'  confidence=0.996  f1=1.000  CORRECT

  COMPARISON   Q: Which factory produced more units in March, Riverside or Lakeside?
               gold='Riverside'
               tok_entropy_norm=0.818  sent_entropy_norm=0.943  question_answer_overlap_coef=1.000
               predicted_answer='Hillcrest'  confidence=0.479  f1=0.000  WRONG

  SUPERLATIVE  Q: Which factory produced the most units in March?
               gold='Riverside'
               tok_entropy_norm=0.865  sent_entropy_norm=0.922  question_answer_overlap_coef=0.000
               predicted_answer='Hillcrest'  confidence=0.876  f1=0.000  WRONG

  deltas vs LITERAL:
    COMPARISON   tok_entropy_norm=+0.070  sent_entropy_norm=+0.475  overlap=+1.000  confidence=-0.516  f1=-1.000
    SUPERLATIVE  tok_entropy_norm=+0.116  sent_entropy_norm=+0.455  overlap=+0.000  confidence=-0.119  f1=-1.000

====================================================================================================
PASSAGE: Sarah and her friend had a marathon yesteday afterwork. Sarah finished the marathon in 3 hours and 45 minutes. Her teammate Priya finished in 3 hours and 52 minutes. Their teammate Wei finished in 3 hours and 38 minutes. The marathon was all first time particpating for all of them.

  LITERAL      Q: How long did Sarah take to finish the marathon?
               gold='3 hours and 45 minutes'
               tok_entropy_norm=0.807  sent_entropy_norm=0.694  question_answer_overlap_coef=0.000
               predicted_answer='3 hours and 45 minutes'  confidence=0.996  f1=1.000  CORRECT

  COMPARISON   Q: Who finished the marathon faster, Sarah or Priya?
               gold='Sarah'
               tok_entropy_norm=0.865  sent_entropy_norm=0.952  question_answer_overlap_coef=1.000
               predicted_answer='Sarah'  confidence=0.984  f1=1.000  CORRECT

  SUPERLATIVE  Q: Who finished the marathon fastest among the three teammates?
               gold='Wei'
               tok_entropy_norm=0.906  sent_entropy_norm=0.988  question_answer_overlap_coef=0.000
               predicted_answer='Priya'  confidence=0.960  f1=0.000  WRONG

  deltas vs LITERAL:
    COMPARISON   tok_entropy_norm=+0.058  sent_entropy_norm=+0.258  overlap=+1.000  confidence=-0.012  f1=+0.000
    SUPERLATIVE  tok_entropy_norm=+0.099  sent_entropy_norm=+0.294  overlap=+0.000  confidence=-0.036  f1=-1.000

====================================================================================================
PASSAGE: Museum Charsm is one of the oldest museums in the country. It had been in private control for many years until 1994. The museum's east wing opened in 1998. The west wing opened in 2006, after a major renovation project. The south wing, the newest addition, opened in 2015. Nowadays, visitors can freely enter and wander all the museum with a very affordable cost of 15e. The museum is open from 9 to 6 everyday except Monday. 

  LITERAL      Q: When did the museum's west wing open?
               gold='2006'
               tok_entropy_norm=0.809  sent_entropy_norm=0.748  question_answer_overlap_coef=0.000
               predicted_answer='2006'  confidence=0.999  f1=1.000  CORRECT

  COMPARISON   Q: Which wing of the museum opened first, the east wing or the west wing?
               gold='the east wing'
               tok_entropy_norm=0.814  sent_entropy_norm=0.796  question_answer_overlap_coef=1.000
               predicted_answer="The museum's east wing"  confidence=0.779  f1=0.857  CORRECT

  SUPERLATIVE  Q: Which wing of the museum opened most recently?
               gold='the south wing'
               tok_entropy_norm=0.881  sent_entropy_norm=0.869  question_answer_overlap_coef=0.500
               predicted_answer='The south wing'  confidence=0.955  f1=1.000  CORRECT

  deltas vs LITERAL:
    COMPARISON   tok_entropy_norm=+0.005  sent_entropy_norm=+0.048  overlap=+1.000  confidence=-0.221  f1=-0.143
    SUPERLATIVE  tok_entropy_norm=+0.072  sent_entropy_norm=+0.120  overlap=+0.500  confidence=-0.045  f1=+0.000

====================================================================================================
PASSAGE: The Zenith X200 smartphone has a battery life of 14 hours and costs $699. Its main competitor, the Aurora S5, has a battery life of 18 hours but costs $799. A newer budget option, the Nova Lite, has a battery life of 20 hours and costs only $549. Reviewers noted that despite its higher price, the Aurora S5 has become popular among frequent travelers who value longer battery life over cost savings. The Zenith X200 remains popular with budget-conscious buyers who charge their phones more frequently throughout the day. The Nova Lite has quickly gained a following among budget shoppers who also want long battery life.

  LITERAL      Q: What is the battery life of the Aurora S5?
               gold='18 hours'
               tok_entropy_norm=0.838  sent_entropy_norm=0.717  question_answer_overlap_coef=0.000
               predicted_answer='18 hours'  confidence=0.997  f1=1.000  CORRECT

  COMPARISON   Q: Which phone has a longer battery life, the Zenith X200 or the Aurora S5?
               gold='the Aurora S5'
               tok_entropy_norm=0.848  sent_entropy_norm=0.844  question_answer_overlap_coef=1.000
               predicted_answer='Zenith X200'  confidence=0.671  f1=0.000  WRONG

  SUPERLATIVE  Q: Which of the three phones has the longest battery life?
               gold='the Nova Lite'
               tok_entropy_norm=0.803  sent_entropy_norm=0.718  question_answer_overlap_coef=0.000
               predicted_answer='Zenith X200'  confidence=0.509  f1=0.000  WRONG

  deltas vs LITERAL:
    COMPARISON   tok_entropy_norm=+0.010  sent_entropy_norm=+0.127  overlap=+1.000  confidence=-0.326  f1=-1.000
    SUPERLATIVE  tok_entropy_norm=-0.035  sent_entropy_norm=+0.001  overlap=+0.000  confidence=-0.488  f1=-1.000

====================================================================================================
PASSAGE: A recent health department report compared emergency room wait times at three hospitals in the same county. At Northside General, the average wait time for non-critical patients was 94 minutes, and the hospital employed 12 full-time emergency physicians. At Southbrook Medical Center, the average wait time was 61 minutes, despite employing only 9 full-time emergency physicians. At Eastview Regional, the average wait time was 108 minutes, the longest of the three, even though the hospital employed 14 full-time emergency physicians. Investigators found that Southbrook's shorter wait times were largely due to a triage software system introduced two years earlier, which allowed nurses to route minor cases to a fast-track unit staffed by physician assistants. Northside had piloted a similar system the previous year but discontinued it after complaints about inconsistent case routing. Eastview has not yet implemented any triage software, and administrators say budget constraints have delayed the project. The health department recommended that Northside and Eastview both revisit the triage software with updated training protocols before their next budget cycles.

  LITERAL      Q: How many full-time emergency physicians did Northside General employ?
               gold='12'
               tok_entropy_norm=0.869  sent_entropy_norm=0.857  question_answer_overlap_coef=0.000
               predicted_answer='12'  confidence=0.997  f1=1.000  CORRECT

  COMPARISON   Q: Which hospital had shorter emergency room wait times, Northside General or Southbrook Medical Center?
               gold='Southbrook Medical Center'
               tok_entropy_norm=0.826  sent_entropy_norm=0.880  question_answer_overlap_coef=1.000
               predicted_answer='Southbrook Medical Center, the average wait time was 61 minutes, despite employing only 9 full-time emergency physicians. At Eastview Regional, the average wait time was 108 minutes, the longest of the three, even though the hospital employed 14 full-time emergency physicians. Investigators found that Southbrook'  confidence=0.456  f1=0.125  WRONG

  SUPERLATIVE  Q: Which of the three hospitals had the longest emergency room wait time?
               gold='Eastview Regional'
               tok_entropy_norm=0.829  sent_entropy_norm=0.813  question_answer_overlap_coef=0.000
               predicted_answer='Eastview Regional'  confidence=1.000  f1=1.000  CORRECT

  deltas vs LITERAL:
    COMPARISON   tok_entropy_norm=-0.043  sent_entropy_norm=+0.023  overlap=+1.000  confidence=-0.541  f1=-0.875
    SUPERLATIVE  tok_entropy_norm=-0.039  sent_entropy_norm=-0.044  overlap=+0.000  confidence=+0.003  f1=+0.000
```

## §8 output — vocabulary/phrasing complexity

```
====================================================================================================
PASSAGE: Lily always carries a spare phone charger in her bag. Her phone battery tends to die quickly during long days at work.

  PLAIN        Q: What does Lily always carry in her bag?
               gold='a spare phone charger'
               tok_entropy_norm=0.848  sent_entropy_norm=0.585  question_answer_overlap_coef=0.000
               predicted_answer='a spare phone charger'  confidence=0.714  f1=1.000  CORRECT

  DENSE        Q: What accessory does Lily routinely transport within her bag?
               gold='a spare phone charger'
               tok_entropy_norm=0.844  sent_entropy_norm=0.648  question_answer_overlap_coef=0.000
               predicted_answer='phone charger'  confidence=0.899  f1=0.667  CORRECT

  delta: tok_entropy_norm=-0.004  sent_entropy_norm=+0.062  overlap=+0.000  confidence=+0.185  f1=-0.333

====================================================================================================
PASSAGE: Every school day, Maria wakes up early. She needs to go to school at 9AM, so she prepares her books and uniform the night before. Her mother drops her off on the way to work.

  PLAIN        Q: What does she need to do at 9AM?
               gold='go to school'
               tok_entropy_norm=0.887  sent_entropy_norm=0.814  question_answer_overlap_coef=0.000
               predicted_answer='go to school'  confidence=0.968  f1=1.000  CORRECT

  DENSE        Q: What obligation necessitates her early departure at 9AM?
               gold='go to school'
               tok_entropy_norm=0.937  sent_entropy_norm=0.909  question_answer_overlap_coef=0.000
               predicted_answer='She needs to go to school'  confidence=0.232  f1=0.667  CORRECT

  delta: tok_entropy_norm=+0.050  sent_entropy_norm=+0.095  overlap=+0.000  confidence=-0.736  f1=-0.333

====================================================================================================
PASSAGE: Every winter, the small town of Millbrook loses power for a few hours during storms. Last year, the local government installed a backup generator at the community center to keep the building running during outages. The generator can supply electricity for up to 12 hours before needing more fuel. Residents can charge their phones and stay warm there when their home power goes out. Some elderly residents rely on the center to keep medical equipment running. The town is now considering installing solar panels to reduce fuel costs.

  PLAIN        Q: What did the local government install at the community center?
               gold='a backup generator'
               tok_entropy_norm=0.843  sent_entropy_norm=0.741  question_answer_overlap_coef=0.000
               predicted_answer='a backup generator'  confidence=0.958  f1=1.000  CORRECT

  DENSE        Q: What apparatus did municipal authorities procure and install at the communal facility?
               gold='a backup generator'
               tok_entropy_norm=0.809  sent_entropy_norm=0.739  question_answer_overlap_coef=0.000
               predicted_answer='backup generator'  confidence=0.871  f1=0.800  CORRECT

  delta: tok_entropy_norm=-0.034  sent_entropy_norm=-0.002  overlap=+0.000  confidence=-0.088  f1=-0.200

====================================================================================================
PASSAGE: In 2015, a small biotech company in Boston began testing a new type of bandage designed for chronic wounds. Unlike traditional bandages, this one contained a thin layer of sensors that measured moisture, temperature, and bacterial growth in real time. The data was sent wirelessly to a smartphone app, allowing doctors to monitor a patient's healing progress without requiring a clinic visit. Early trials showed that patients using the smart bandage healed nearly 20 percent faster than those using standard dressings. Doctors were able to adjust treatment plans within a day of an infection appearing, rather than waiting for the patient's next scheduled appointment. The bandage costs significantly more than a regular one, which has slowed its adoption in hospitals with tight budgets. Even so, several insurance companies have begun covering the cost for patients with diabetes, who are especially prone to slow-healing wounds. The company hopes to expand production and lower the price within the next two years.

  PLAIN        Q: What three things did the sensors in the bandage measure?
               gold='moisture, temperature, and bacterial growth'
               tok_entropy_norm=0.868  sent_entropy_norm=0.828  question_answer_overlap_coef=0.000
               predicted_answer='moisture, temperature, and bacterial growth'  confidence=0.997  f1=1.000  CORRECT

  DENSE        Q: What three physiological parameters were quantified by the embedded sensor array?
               gold='moisture, temperature, and bacterial growth'
               tok_entropy_norm=0.881  sent_entropy_norm=0.838  question_answer_overlap_coef=0.000
               predicted_answer='moisture, temperature, and bacterial growth'  confidence=0.998  f1=1.000  CORRECT

  delta: tok_entropy_norm=+0.012  sent_entropy_norm=+0.009  overlap=+0.000  confidence=+0.001  f1=+0.000

====================================================================================================
PASSAGE: When Elena took over as principal of Riverside Elementary three years ago, nearly a third of the school's students were missing more than two weeks of class every year. After interviewing families, she discovered that many parents worked early shifts and had no way to get their children to school on time, since the nearest bus stop was almost two miles away. Elena partnered with a local transit company to add a new bus route that stopped directly in front of the school at 7:15 each morning. She also arranged for breakfast to be served starting at 7:00, so students arriving early would not have to wait outside in the cold. Within the first semester, chronic absenteeism dropped from 32 percent to 19 percent. Teachers reported that students who used to miss the first class of the day were now consistently present, which allowed lessons to build on each other more smoothly. The district was impressed enough that it approved funding to extend the new bus route to two neighboring schools the following year. Some parents initially worried that the earlier start time would be difficult for younger children, but a survey conducted at the end of the year found that most families adjusted within the first month. Elena says the biggest lesson from the program was that a single logistical barrier, transportation, was quietly undermining years of curriculum improvements the school had already made.

  PLAIN        Q: What time does the new bus route stop in front of the school?
               gold='7:15'
               tok_entropy_norm=0.844  sent_entropy_norm=0.791  question_answer_overlap_coef=0.000
               predicted_answer='7:15'  confidence=0.999  f1=1.000  CORRECT

  DENSE        Q: At what hour does the newly instituted transit route arrive at the institution's entrance?
               gold='7:15'
               tok_entropy_norm=0.855  sent_entropy_norm=0.837  question_answer_overlap_coef=0.000
               predicted_answer='7:15'  confidence=1.000  f1=1.000  CORRECT

  delta: tok_entropy_norm=+0.011  sent_entropy_norm=+0.047  overlap=+0.000  confidence=+0.001  f1=+0.000
```
