# Entropy on captured-correct samples only: `consciousAI/question-answering-roberta-base-s-v2`

layer=11, target 6 captured_correct passages per group (up to 400 candidates tried per group, up to 5 questions shown per passage). `captured_correct` = f1>=0.5 OR (recall_overlap>=0.8 AND pred<= 30% of passage). A passage is included if at least one of its real questions passes the gate -- entropy is only a trustworthy difficulty signal on rows where captured=Y. `combined` is the simple average of sent_entropy_norm and tok_entropy_norm. Each passage also gets a fixed topic-probe question ("What is the main topic of the passage?") for comparison -- it has no gold answer, so no f1/captured_correct for it.

## Summary: avg entropy by group (captured_correct questions only)

| source | level | sent_entropy_norm | tok_entropy_norm | combined | probe combined | ans_sentence_share | ans_sentence_rank | n_questions | n_passages |
|---|---|---|---|---|---|---|---|---|---|
| RACE-middle | EASY | 0.816 | 0.641 | 0.728 | 0.739 | 0.083 | 7.62 | 8 | 6 |
| RACE-high | MEDIUM | 0.870 | 0.661 | 0.766 | 0.759 | 0.081 | 7.33 | 6 | 6 |
| RACE-C | HARD | 0.858 | 0.657 | 0.757 | 0.742 | 0.143 | 3.14 | 7 | 6 |
| OneStopQA | EASY | 0.884 | 0.696 | 0.790 | 0.700 | 0.182 | 3.67 | 6 | 6 |
| OneStopQA | MEDIUM | 0.830 | 0.606 | 0.718 | 0.651 | 0.297 | 1.67 | 6 | 6 |
| OneStopQA | HARD | 0.852 | 0.656 | 0.754 | 0.700 | 0.138 | 4.00 | 7 | 6 |
| SQuAD | N/A | 0.884 | 0.621 | 0.753 | 0.674 | 0.230 | 3.00 | 23 | 6 |

## Per-passage detail: RACE-C / HARD (avg over each passage's captured_correct questions)

`ans_sentence_share` = avg fraction of attention mass on the sentence containing the model's own predicted answer; `ans_sentence_rank` = avg rank of that sentence by attention mass (1 = most-attended sentence IS the answer sentence -- lower is "more correctly focused").

| passage | snippet | n_captured/n_total | sent_entropy_norm | tok_entropy_norm | combined | probe combined | ans_sentence_share | ans_sentence_rank |
|---|---|---|---|---|---|---|---|---|
| 1 | Vitamins are important to our health.Difierent vitamins are found in d... | 1/4 | 0.700 | 0.530 | 0.615 | 0.657 | 0.045 | 8.00 |
| 2 | Anne Whitney, a sophomore at Colorado State University, first had a pr... | 1/5 | 0.757 | 0.603 | 0.680 | 0.708 | 0.219 | 1.00 |
| 3 | Health Minister Nicola Roxon's latest proposal that patients be alloca... | 1/3 | 0.812 | 0.715 | 0.763 | 0.701 | 0.166 | 1.00 |
| 4 | The English policeman has several nicknames but the most frequently us... | 1/4 | 0.924 | 0.668 | 0.796 | 0.768 | 0.047 | 9.00 |
| 5 | As supplier of most of the food we eat and of raw materials for many i... | 2/5 | 0.939 | 0.685 | 0.812 | 0.823 | 0.185 | 1.00 |
| 6 | The biggest safety threat facing airlines today may not be a terrorist... | 1/5 | 0.932 | 0.714 | 0.823 | 0.797 | 0.152 | 1.00 |

## RACE-middle / EASY

6/6 captured (tried 7 candidates)

### Passage 1

**Passage:**

> When travelling.you are sure to try some exciting new kinds of food.The Wild Food Festival,in the town of Hokitika,the west of Coast of New Zealand,gives you the chance to try some strange food.It is a celebration of the areas special lifestyle and food.And it celebrates food that most people might not want to eat.It is held in March every year.
At the festival you will find huhu grubs and beetles on your plate.The festival also celebrates Maori food. the food of the traditional native people of New Nealand And visitors will eat the wild food with plenty of famous West Coast beer.What's more,there are three stages at the festival,where there is live music and entertainment an day long.
If you have the chance to travel to Hokitita during the Wild Food Festival,you should book a hotel before it begins.or you can choose to stay at local schools.A number of local schools become camping grounds over the weekend of the festival.You can also stay in Greymouth,because there are buses from Greymouth to the festival.

(2/4 of this passage's questions were captured_correct -- only those are shown below)

**Question 1:** When is the Wild Food Festival held in the town of Hokitika every year?

**Gold answer:** In March.

**Predicted answer:** March (f1=0.67, conf=0.908)

**Entropy:** sent_entropy_norm=0.798, tok_entropy_norm=0.595, combined=0.697 (sent_entropy=2.107, tok_entropy=3.225, num_sentences=14)

**Answer-sentence attention:** share=0.190, rank=3/14 -- how much attention mass landed on the sentence containing the model's own predicted answer, and its rank among all sentences (1 = most-attended sentence IS the answer sentence).

**Question 2:** What will you find on your plate at the festival?

**Gold answer:** Huhu grubs and beetles.

**Predicted answer:** huhu grubs and beetles (f1=1.00, conf=0.976)

**Entropy:** sent_entropy_norm=0.784, tok_entropy_norm=0.655, combined=0.719 (sent_entropy=2.068, tok_entropy=3.548, num_sentences=14)

**Answer-sentence attention:** share=0.090, rank=4/14 -- how much attention mass landed on the sentence containing the model's own predicted answer, and its rank among all sentences (1 = most-attended sentence IS the answer sentence).

**Probe question:** What is the main topic of the passage?

**Probe predicted answer:** food (conf=0.005)

**Probe entropy:** sent_entropy_norm=0.779, tok_entropy_norm=0.588, combined=0.684

### Passage 2

**Passage:**

> George Stephenson was born in 1781 in a poor family. He had to start work when he was only eight.When George was fourteen, he became his father's helper.He spent a lot of time learning about engines .And on holidays he often made one engine to pieces and studied each piece carefully.Soon he became a very good worker though he could not read or write.He began to learn English letters when he was seventeen years old.Every day after he did twelve hours of hard work, he walked a long way to have lessons from a young school teacher. On his eighteenth birthday,he wrote his own name for the first time in his life.George invented  many things in his life.The train was the greatest one among them.Today when we take trains from one place to another,we'll think of this great man---George Stephenson.

(2/5 of this passage's questions were captured_correct -- only those are shown below)

**Question 1:** He spent a lot of time learning about engines and soon he became   _  .

**Gold answer:** a good worker

**Predicted answer:** a very good worker (f1=0.86, conf=0.622)

**Entropy:** sent_entropy_norm=0.787, tok_entropy_norm=0.574, combined=0.680 (sent_entropy=1.955, tok_entropy=2.953, num_sentences=12)

**Answer-sentence attention:** share=0.070, rank=5/12 -- how much attention mass landed on the sentence containing the model's own predicted answer, and its rank among all sentences (1 = most-attended sentence IS the answer sentence).

**Question 2:** George Stephenson invented the   _  .

**Gold answer:** the train

**Predicted answer:** The train (f1=1.00, conf=0.905)

**Entropy:** sent_entropy_norm=0.664, tok_entropy_norm=0.498, combined=0.581 (sent_entropy=1.649, tok_entropy=2.564, num_sentences=12)

**Answer-sentence attention:** share=0.075, rank=4/12 -- how much attention mass landed on the sentence containing the model's own predicted answer, and its rank among all sentences (1 = most-attended sentence IS the answer sentence).

**Probe question:** What is the main topic of the passage?

**Probe predicted answer:** engines (conf=0.520)

**Probe entropy:** sent_entropy_norm=0.720, tok_entropy_norm=0.577, combined=0.649

### Passage 3

**Passage:**

> Some people think only school children do not agree with their parents, however, it is not true. Communication is a problem for parents and children of all ages. If it's hard for you to communicate with your parents, don't worry about it. Here are some advice for you to _ the generation gap  .
Don't argue with your parents. Don't get to your parents when you are angry. Your parents probably won't consider your ideas if you are shouting at them. And you can't express yourself well if you are angry. Go someplace to cool off. Make sure you understand why you are unhappy. Then think about what you want to say to your parents. If you don't think you can speak to them at the moment, try writing a letter.
Try to reach a compromise  . Perhaps you and your parents disagree on something. You can keep your disagreement and try your best to accept each other. Michael's mother didn't agree with him about buying a motorbike. They argued over it. But they finally came to a compromise. Michael bought the motorbike, but only drove it on certain days.
Of course, your parents might refuse to compromise on something. In these situations, it is especially important to show love and respect to them. Showing respect will keep your relationship  strong.
Talk about your values. The values of your parents are probably different from those of your own. Tell your parents what you care about, and why. Understanding your values might help them see your purposes in life.
A good relationship with your parents can make you a better and happier person. It is worth having a try!
,.

(1/4 of this passage's questions were captured_correct -- only those are shown below)

**Question 1:** According to the passage who have a communication problem?

**Gold answer:** parents and children of all ages

**Predicted answer:** parents and children of all ages (f1=1.00, conf=0.864)

**Entropy:** sent_entropy_norm=0.873, tok_entropy_norm=0.757, combined=0.815 (sent_entropy=2.939, tok_entropy=4.403, num_sentences=29)

**Answer-sentence attention:** share=0.001, rank=29/29 -- how much attention mass landed on the sentence containing the model's own predicted answer, and its rank among all sentences (1 = most-attended sentence IS the answer sentence).

**Probe question:** What is the main topic of the passage?

**Probe predicted answer:** Communication (conf=0.411)

**Probe entropy:** sent_entropy_norm=0.845, tok_entropy_norm=0.702, combined=0.773

### Passage 4

**Passage:**

> Inventions named after people
Many new things are invented each year. Interestingly, some inventions are named after the people who invented them, making the inventions and inventors easier to be remembered.
The bowler hat is named after London hat-makers Thomas and William Bowler. The brothers received an order from Edward Coke, the younger brother of the 2ndEarl of Leicester. They were asked to design a hat for Coke's gamekeepers to protect _ heads from branches while on horseback. Later, the stylish hat became popular in Europe and the United States.
Another invention that is named after its inventor is Braille, a writing system used by blind people. French educator Louis Braille developed a new system of reading and writing after learning the cryptography of French Captain Charles Barbier during the war. The captain had come up with a code of dots on paper that allowed soldiers to communicate in the dark.
The diesel engine is also named after its inventor----German engineer Rudolf Diesel. After a few dangerous tests, he invented a new and more efficient engine in 1892 and the engine was later called the diesel engine. The engines were widely used in buses, trucks, trains and ships, and Rudolf Diesel became a millionaire.

(1/4 of this passage's questions were captured_correct -- only those are shown below)

**Question 1:** Who invented the bowler hat?

**Gold answer:** Thomas and William Bowler.

**Predicted answer:** Thomas and William Bowler (f1=1.00, conf=0.906)

**Entropy:** sent_entropy_norm=0.981, tok_entropy_norm=0.756, combined=0.868 (sent_entropy=2.438, tok_entropy=4.169, num_sentences=12)

**Answer-sentence attention:** share=0.095, rank=4/12 -- how much attention mass landed on the sentence containing the model's own predicted answer, and its rank among all sentences (1 = most-attended sentence IS the answer sentence).

**Probe question:** What is the main topic of the passage?

**Probe predicted answer:** making the inventions and inventors easier to be remembered (conf=0.065)

**Probe entropy:** sent_entropy_norm=0.936, tok_entropy_norm=0.663, combined=0.800

### Passage 5

**Passage:**

> Whenever there's a terrible storm, there are plenty of sad stories. Many people lost their houses, their cars and their pets. But sometimes these stories have happy endings too.
A family in New Jersey, US, had a cat named Vivien. She is very smart. She could even draw with her wet paws on the floor. They all love her very much. When hurricane   Sandy hit America in October, they moved to a safer place-13km away from home. Sadly Vivien went missing. The whole family were worried about her. They put up posters on the Internet to look for her. All the nine family members searched wherever they thought she could stay, but they didn't find it. The family thought Vivien was gone forever.
But six months later, Vivien showed up at their house, according to Yahoo News. They considered her return as a wonder. No one can be sure where Vivien was for all that time.
Many animals are good at finding their way home. People usually say that dogs and cats find their way home through using their sense of smell.
But that doesn't explain how Vivien found her way back. Hurricane Sandy blew away the normal smells of home.
"I wish she could talk," said her owner.

(1/3 of this passage's questions were captured_correct -- only those are shown below)

**Question 1:** From the passage we know Sandy is a

**Gold answer:** hurricane

**Predicted answer:** hurricane (f1=1.00, conf=0.160)

**Entropy:** sent_entropy_norm=0.833, tok_entropy_norm=0.622, combined=0.728 (sent_entropy=2.536, tok_entropy=3.457, num_sentences=21)

**Answer-sentence attention:** share=0.119, rank=2/21 -- how much attention mass landed on the sentence containing the model's own predicted answer, and its rank among all sentences (1 = most-attended sentence IS the answer sentence).

**Probe question:** What is the main topic of the passage?

**Probe predicted answer:** sad stories (conf=0.312)

**Probe entropy:** sent_entropy_norm=0.878, tok_entropy_norm=0.702, combined=0.790

### Passage 6

**Passage:**

> Dear Jenny,
I'm sorry you're sick and can't come to school this week. Let me tell you what you have on Monday next week.
We have math at 8:00. How boring! Then we have English. That's interesting. I know you love English. Next is P.E.. The fourth lesson is art. That's my favorite subject! In the afternoon, we have history with Mr. Chen. He is fun but history isn't.  Then at 2:00 we have science with Miss Jones. You know how strict she is. I don't like her. Our last class of the day is math.
Yours,
Ben

(1/5 of this passage's questions were captured_correct -- only those are shown below)

**Question 1:** Their first lesson is   _  .

**Gold answer:** math

**Predicted answer:** math (f1=1.00, conf=0.063)

**Entropy:** sent_entropy_norm=0.806, tok_entropy_norm=0.668, combined=0.737 (sent_entropy=2.374, tok_entropy=3.288, num_sentences=19)

**Answer-sentence attention:** share=0.026, rank=10/19 -- how much attention mass landed on the sentence containing the model's own predicted answer, and its rank among all sentences (1 = most-attended sentence IS the answer sentence).

**Probe question:** What is the main topic of the passage?

**Probe predicted answer:** math (conf=0.159)

**Probe entropy:** sent_entropy_norm=0.788, tok_entropy_norm=0.683, combined=0.736

Avg over 8 captured=Y question(s): sent_entropy_norm=0.816, tok_entropy_norm=0.641, combined=0.728 (probe combined avg=0.739); answer_sentence_attention_share=0.083, answer_sentence_rank=7.62 (n_located=8)

## RACE-high / MEDIUM

6/6 captured (tried 24 candidates)

### Passage 1

**Passage:**

> The Sapporo Snow Festival (Sapporo Yuki-matsuri) is a festival being held every year in Sapporo, Japan for over seven days in February. Presently, Odori Park, Susukino, and Tsudome are the main places of the festival. The 2013 Yuki-matsuri dates are February 5 to 11. 
    The festival is one of Japan's largest and most typical winter events. In 2007 (58th Festival), about two million people visited Sapporo to see the hundreds of floating statues and ice sculptures at the Odori Park and Susukino sites, in central Sapporo, and at the Satoland site. The festival is thought to be a chance for promoting international relations. International Snow Sculpture Contest has been held at the Odori Park site since 1974, and 14 teams from various areas of the world participated in 2008. 
   The subject of the statues differs and often shows an event, famous building or person from the previous year. For example, in 2004, there were statues of Hideki Matsui, the famous baseball player who at that time played for the New York Yankees. A number of stages made out of snow are also constructed and some events including musical performances are held. At the Satoland site, visitors can enjoy long snow and ice slides as well as a huge maze  made of snow. Visitors can also enjoy a variety of local foods from all over Hokkaido at the Odori Park and Satoland sites, such as fresh seafood, potatoes and corn, and fresh dairy products. 
    Every year the number of Statues displayed is around 400 in total. In 2007, ther were 307 statues created in the Odori Park site, 32 in the Satoland site and 100 in the Susukino site. The best place to view the creations is from the TV Tower at the Odori Park site. Most of the statues are lighted in the evening. The Sapporo Snow Festival Museum is placed in the Hitsujigaoka observation hill  in Toyohira-ku, and displays historical materials and media of the festival.

(1/3 of this passage's questions were captured_correct -- only those are shown below)

**Question 1:** The Sapporo Snow Festival 2013 will start from   _  .

**Gold answer:** February 5

**Predicted answer:** February (f1=0.67, conf=0.261)

**Entropy:** sent_entropy_norm=0.898, tok_entropy_norm=0.556, combined=0.727 (sent_entropy=2.543, tok_entropy=3.382, num_sentences=17)

**Answer-sentence attention:** share=0.028, rank=12/17 -- how much attention mass landed on the sentence containing the model's own predicted answer, and its rank among all sentences (1 = most-attended sentence IS the answer sentence).

**Probe question:** What is the main topic of the passage?

**Probe predicted answer:** promoting international relations (conf=0.593)

**Probe entropy:** sent_entropy_norm=0.902, tok_entropy_norm=0.693, combined=0.798

### Passage 2

**Passage:**

> SHANGHAI, June 7(AP)--A 16-year-old girl's suicide after she was barred from a key exam draw attention to increasing worries over academic pressures, as millions of Chinese students began annual college entrance tests on Wednesday. 
The three-day exam, viewed as important to future career and financial success, has a record 9.5 million high school students across prefix = st1 /Chinacompeting for just 2.6 million university places. For kids and parents alike, it's a difficulty that experts say causes extreme emotional distress. "Pressure from study and exams is a top reason for psychological problems among Chinese youth," said Jin Wuguan, director of the Youth Psychological Counseling Center at Shanghai'sRuijinHospital. 
In China's increasingly success oriented, pressure-cooker cities, academic stress is seen as a rising cause of youth suicides and even murders of parents by children who are driven crazy by intolerable pressure to perform. 
According to her family and newspaper accounts, 16-year-old Wu Wenwen drowned herself after she was stopped at the exam room door because her hair wasn't tied back as her school required. Returning in tied hair, she was then told the end-of-term exam had already started and she was too late to take it. In tears, Wu called her mother, and then disappeared. Her body was found the same night in a nearby lake. 
China doesn't keep comprehensive statistics on student suicides, but Jin said health care professionals see the problem worsening, even among elementary students. Most Chinese schools still lack advisers and teachers receive little training in spotting symptoms of emotional distress, Jin said. Parents are little help, often piling on pressure while ignoring their children's emotional development, he said. "It's a basic unwillingness or inability to recognize and deal with with emotional problems," Jin said. 
Wang Yufeng, of Peking University's Institute of Mental, estimates the rate of emotional disorders such as depression among Chinese students under age 17 at up to 32 percent , a total of 30 million students. 
Others say that figure may be as high as 50 percent. A survey last year by the government's China Youth and ChildrenResearchCentershowed 57.6 percent of students felt highly distressed by academic pressures.

(1/3 of this passage's questions were captured_correct -- only those are shown below)

**Question 1:** Where will we most probably find the article?

**Gold answer:** In a newspaper.

**Predicted answer:** newspaper (f1=0.50, conf=0.048)

**Entropy:** sent_entropy_norm=0.901, tok_entropy_norm=0.720, combined=0.811 (sent_entropy=2.653, tok_entropy=4.433, num_sentences=19)

**Answer-sentence attention:** share=0.056, rank=12/19 -- how much attention mass landed on the sentence containing the model's own predicted answer, and its rank among all sentences (1 = most-attended sentence IS the answer sentence).

**Probe question:** What is the main topic of the passage?

**Probe predicted answer:** academic pressures (conf=0.895)

**Probe entropy:** sent_entropy_norm=0.872, tok_entropy_norm=0.689, combined=0.781

### Passage 3

**Passage:**

> The top ten hottest English words in 2015
Selfie   is "taking a photo by yourself", Now "taking a selfie" is generally a way of self-expression.
Budgetwife is the opposite of budget husband  . By name, you can see that the economy strength of budget male is not strong as the "diamond man"  , but he is both economically and emotionally reliable.
Phubbing refers to impolite behavior that in social situations people don't pay attention to the people around, but just look at their mobile phones, we can call it "down". People are called phubber "down"  .
Bromeo   are male girlfriends. He is one of your most loyal friends, will support you in every situation.
"Fangirl" or "fanboy"   refers to those who are crazy about something or a star, even to the point of sanity.
Gayriage   refers to two people of equal gender form of marriage. Two men to get married, the marriage is called gayriage.
Mompetition, it is the competition between mothers, comparing whose child is more beautiful, more smarter, more fashionable. It can be compared two or more mothers, and the children being compared can be adult.
Social bubble  , which means that some people seem to know many people, but only few people could be friends. After "financial bubble", "housing bubble", personal bubbles begin to hit career people.

(1/3 of this passage's questions were captured_correct -- only those are shown below)

**Question 1:** Tom is a   _  , crazy about selfies.

**Gold answer:** fanboy

**Predicted answer:** Fangirl" or "fanboy (f1=0.50, conf=0.015)

**Entropy:** sent_entropy_norm=0.873, tok_entropy_norm=0.667, combined=0.770 (sent_entropy=2.303, tok_entropy=3.809, num_sentences=14)

**Answer-sentence attention:** share=0.154, rank=2/14 -- how much attention mass landed on the sentence containing the model's own predicted answer, and its rank among all sentences (1 = most-attended sentence IS the answer sentence).

**Probe question:** What is the main topic of the passage?

**Probe predicted answer:** M (conf=0.000)

**Probe entropy:** sent_entropy_norm=0.876, tok_entropy_norm=0.592, combined=0.734

### Passage 4

**Passage:**

> What's on TV?
6:00  Channel 3Let's Talk! Guest: Animal expert Jim Porter
      Channel 5   Cartoons
      Channel 8   News
      Channel 9   News
7:00      Channel 3Cooking with Cathy Tonight: Chicken with mushrooms
      Channel 5   MovieA laugh a Minute(1955) James Rayburn
      Channel 8   Spin   for Dollars!
      Channel 9   Farm Report
7:30      Channel 3   Double Trouble (comedy)The twins disrupt the high school dance
      Channel 9   Wall Street Today. Stock Market Report
8:00      Channel 3   NBA Basketball. Teams to be announced
Channel 8  Movie At Day's End (1981) Michael Collier, Julie Romer. Drama set in World War II
Channel 9   News Special "Saving Our Waterways: Pollution in the Mississippi"

(1/2 of this passage's questions were captured_correct -- only those are shown below)

**Question 1:** Which is most probably the News Channel?

**Gold answer:** Channel9.

**Predicted answer:** Channel 9 (f1=0.00, conf=0.358)

**Entropy:** sent_entropy_norm=0.751, tok_entropy_norm=0.547, combined=0.649 (sent_entropy=1.461, tok_entropy=2.985, num_sentences=7)

**Answer-sentence attention:** share=0.033, rank=5/7 -- how much attention mass landed on the sentence containing the model's own predicted answer, and its rank among all sentences (1 = most-attended sentence IS the answer sentence).

**Probe question:** What is the main topic of the passage?

**Probe predicted answer:** Pollution in the Mississippi (conf=0.413)

**Probe entropy:** sent_entropy_norm=0.792, tok_entropy_norm=0.584, combined=0.688

### Passage 5

**Passage:**

> The iPhone, the iPad, each of Apple's products sounds cool and has become a fad . Apple has cleverly taken advantage of the power of the letter "i" - and many other brands are following suit. The BBC's iPlayer - which allows Web users to watch TV programs on the Internet -adopted the title in 2008. A lovely bear - popular in the US and UK - that plays music and video is called "iTeddy". A slimmed-down version of London's Independent newspaper was launched last week under the name "i".
In general, single-letter prefixes have been popular since the 1990s, when terms such as e-mail and e-commerce first came into use. 
Most "i" products are targeted at young people and considering the major readers of Independent's "i", it's no surprise that they've selected this fashionable name. 
But it's hard to see what's so special about the letter "i". Why not use "a", "b", or "c" instead? According to Tony Thorne, head of the Language Center at King's College, London, "i" works because its meaning has become ambiguous. When Apple uses "i", no one knows whether it means Internet, information, individual or interactive, Thorne told BBC Magazines. "Even when Apple created the iPod, it seems it didn't have one clear definition," he says. 
"However, thanks to Apple, the term is now associated with portability." adds Thorne.
Clearly the letter "i" also agrees with the idea that the Western World is centered on the individual. Each person believes they have their own needs, and we love personalized products for this reason. 
Along with "Google" and "blog", readers of BBC Magazines voted "i" as one of the top 20 words that have come to define the last decade. 
But as history shows, people grow tired of fads. From the 1900s to 1990s, products with "2000" in their names became fashionable as the year was associated with all things advanced and modern. However, as we entered the new century, the trend inevitably  disappeared.

(1/4 of this passage's questions were captured_correct -- only those are shown below)

**Question 1:** We can infer that the Independent's "i" is aimed at   _  .

**Gold answer:** young readers

**Predicted answer:** young people (f1=0.50, conf=0.945)

**Entropy:** sent_entropy_norm=0.865, tok_entropy_norm=0.658, combined=0.762 (sent_entropy=2.592, tok_entropy=4.018, num_sentences=20)

**Answer-sentence attention:** share=0.171, rank=1/20 -- how much attention mass landed on the sentence containing the model's own predicted answer, and its rank among all sentences (1 = most-attended sentence IS the answer sentence).

**Probe question:** What is the main topic of the passage?

**Probe predicted answer:** Most "i" products are targeted at young people and considering the major readers of Independent's "i", it's no surprise that they've selected this fashionable name. 
But it's hard to see what's so special about the letter "i". Why not use "a", "b", or "c" instead? According to Tony Thorne, head of the Language Center at King's College, London, "i" works because its meaning has become ambiguous. When Apple uses "i", no one knows whether it means Internet, information, individual or interactive, Thorne told BBC Magazines. "Even when Apple created the iPod, it seems it didn't have one clear definition," he says. 
"However, thanks to Apple, the term is now associated with portability." adds Thorne.
Clearly the letter "i" also agrees with the idea that the Western World is centered on the individual. Each person believes they have their own needs (conf=0.004)

**Probe entropy:** sent_entropy_norm=0.877, tok_entropy_norm=0.623, combined=0.750

### Passage 6

**Passage:**

> Our flat was on the fifth floor but you could still hear the roar of the ocean and see the stars at night. I used to take long walks along the water. The food in town was delicious and the people were very friendly. The area was very quiet and peaceful, and fairly deserted. 
The last evening of our vacation, however, we all heard strange footsteps following closely behind us as we were walking up to our flat in the holiday centre. We turned around and noticed a fairly young man moving very rapidly across the beach and getting closer to us. He was tall and wore a baseball cap. We couldn't see his face and he was approaching us very rapidly. The man's actions made my dad very nervous. Dad warned us that we'd better try to make it to our flat as quickly as possible. I didn't like my dad's voice; I could hear fear in it. It was late and we were all alone. We didn't have any cell phones on us. I never saw Dad as worried as he was then and I knew that something was terribly wrong. The sense of fear started to overwhelm Mom and me. We had had such a good time in town. Now, the night was rapidly turning into a dangerous situation. 
We could hear the man's footsteps getting closer. Dad's face was almost pale. The so-called intruder   had moved nearer and nearer when all of a sudden, the nearby vending  machine started going crazy and spitting out cans of soda! The noise actually scared the intruder and he ran out of sight. My parents were shaking, but we all turned around to see who had put money into the vending machine downstairs, and actually saved us, but no one was around at all. Not a soul. 
It's one vacation I will never forget.

(1/2 of this passage's questions were captured_correct -- only those are shown below)

**Question 1:** What helped them get rid of the trouble?

**Gold answer:** The noise from the vending machine.

**Predicted answer:** soda! The noise actually scared the intruder and he ran out of sight. My parents were shaking, but we all turned around to see who had put money into the vending machine (f1=0.21, conf=0.000)

**Entropy:** sent_entropy_norm=0.935, tok_entropy_norm=0.819, combined=0.877 (sent_entropy=2.973, tok_entropy=4.835, num_sentences=24)

**Answer-sentence attention:** share=0.042, rank=12/24 -- how much attention mass landed on the sentence containing the model's own predicted answer, and its rank among all sentences (1 = most-attended sentence IS the answer sentence).

**Probe question:** What is the main topic of the passage?

**Probe predicted answer:** water (conf=0.001)

**Probe entropy:** sent_entropy_norm=0.888, tok_entropy_norm=0.723, combined=0.805

Avg over 6 captured=Y question(s): sent_entropy_norm=0.870, tok_entropy_norm=0.661, combined=0.766 (probe combined avg=0.759); answer_sentence_attention_share=0.081, answer_sentence_rank=7.33 (n_located=6)

## RACE-C / HARD

6/6 captured (tried 42 candidates)

### Passage 1

**Passage:**

> Vitamins are important to our health.Difierent vitamins are found in difierent
foods.grains.vegetables and fruits,fish and meat,eggs and milk products.So which foods should  be eaten to get enough of the vitamins our bodies need?Let us look at some important vitamins for the answer.
Vitamin A helps prevent skin and other tissues from becoming dry.People who do not get enough Vitamin A cannot see well in darkness.They may develop a condition that dries the eyes.This can result in infections and lead to blindness.Vitamin A is found in fish liver oil.It
is also in the yellow part of eggs.Sweet potatoes.carrots and other darkly colored fruits and vegetables contain substances that the body can change into Vitamin A.
Vitamin B is also called thiamine.Thiamine changes starchy foods jnto energy.It also
helps the heart and nervous system work smoothly.Without it.we would be weak and would not grow.We also might develop beriberi.Thiamine is found not iust in whole grains like brown rice.but also jn other foods.These include beans and peas.nuts.and meat and fish.
Vitamin C is needed for strong bones and teeth.and for healthy blood passages.It also helps wounds heal quickly.The body stores little Vitamin C.So we must get it every day in foods such as citrus fruits,tomatoes and uncooked cabbage.
Vitamin D increases levels of the element calcium(')in the blood.Calcium is needed for nerve and muscle cells to work normally.It is also needed to build strong bones.VitaminD prevents the children's bone disease rickets.Ultraviolet light from the sun changes a substance in the skin into vitamin D Fish liver oil also contains vitamin D.In some countries.milk producers add vitamin D to milk so children will get enough.
Vitamin K is needed for healthy blood.It thickens the blood around a cut to stop bleeding.Bacteria in the intestines f)normally produce vitamin K.It can also be found in pork products.1iver and in vegetables like cabbage.kale and spinach.

(1/4 of this passage's questions were captured_correct -- only those are shown below)

**Question 1:** Lack of Vitamin A will lead to _ .

**Gold answer:** night blindness

**Predicted answer:** blindness (f1=0.67, conf=0.854)

**Entropy:** sent_entropy_norm=0.700, tok_entropy_norm=0.530, combined=0.615 (sent_entropy=2.634, tok_entropy=3.249, num_sentences=43)

**Answer-sentence attention:** share=0.045, rank=8/43 -- how much attention mass landed on the sentence containing the model's own predicted answer, and its rank among all sentences (1 = most-attended sentence IS the answer sentence).

**Probe question:** What is the main topic of the passage?

**Probe predicted answer:** Vitamins are important to our health (conf=0.262)

**Probe entropy:** sent_entropy_norm=0.730, tok_entropy_norm=0.584, combined=0.657

### Passage 2

**Passage:**

> Anne Whitney, a sophomore at Colorado State University, first had a problem taking tests when she began college. "I was always well prepared for my tests. Sometimes I studied for weeks before a test. Yet I would go in to take the test, only to find I could not answer the questions correctly. I would blank out because of nervousness and fear. I couldn't think of the answer. My low grades on the tests did not show what I knew to the teacher." Another student in microbiology and similar experiences. He said, "My first chemistry test was very difficult. Then, on the second test, sat down to take it, and I was so nervous that I was shaking. My hands were moving up and down so quickly that it was hard to hold my pencil. I knew the materical and I knew the answers. Yet I couldn't even writen them down!"
These two young students were experiencing something called test anxiety. Because a student worries and it uneasy about a test, his or her mind does not work as well as it usually does. The student can't write or think clearly because of the extreme tension and nervousness. Although poor grades are often a result of poor study habits, sometimes test anxiety causes the low grades. Recently, test anxiety had been recognized as a real problem, not just an excuse or a false explanation of lazy students.
Special university counseling courses try to help students. In these courses, counselors try to help students by teaching them how to manage test anxiety. At some universities, students take tests to measure their anxiety. If the tests show their anxiety is high, the students can take short courses to help them deal with their tension. These courses teach students how to relax their bodies. Students are trained to become calm in bery tense situations. By controlling their nervousness, they can let their minds work at ease. Learned information then comes out without difficulty on a test.
An expert at the University of California explains. "With almost all students, relaxation and less stress are felt after taking out program. Most of then experience better control during their tests. Almost all have some improvement. With some, the improvement is very great."

(1/5 of this passage's questions were captured_correct -- only those are shown below)

**Question 1:** Test anxiety has been recognized as _ .

**Gold answer:** a real problem

**Predicted answer:** a real problem (f1=1.00, conf=0.496)

**Entropy:** sent_entropy_norm=0.757, tok_entropy_norm=0.603, combined=0.680 (sent_entropy=2.623, tok_entropy=3.670, num_sentences=32)

**Answer-sentence attention:** share=0.219, rank=1/32 -- how much attention mass landed on the sentence containing the model's own predicted answer, and its rank among all sentences (1 = most-attended sentence IS the answer sentence).

**Probe question:** What is the main topic of the passage?

**Probe predicted answer:** chemistry (conf=0.422)

**Probe entropy:** sent_entropy_norm=0.805, tok_entropy_norm=0.611, combined=0.708

### Passage 3

**Passage:**

> Health Minister Nicola Roxon's latest proposal that patients be allocated to doctors on a list basis is straight out of the playbook of Britain's National Health Service.
Let's think about this from the patient's point of view. Some doctors are better than others, the same as some plumbers are better than others. The reason may be a better bedside manner; it may be they are more competent; it may be just that there is a simple personality clash-it may just be that, at times, the patient wants a second opinion. Or it might be that the patient has a potentially embarrassing problem that he or she does not want to discuss with his or her regular general practitioner.
Some people who are ill-suited to their career choice are always going to slip through the system. In other words, if you are allocated a doctor you don't like or who is a dud, you are likely to be stuck with him. Of course, the government will make some noises about "freedom of choice"; but in the end, a doctor who hangs up his shingle and succeeds or fails according to the quality of service he offers is going to provide a better quality of service than a public employee.
Now, all doctors, including general practitioners, must be members of the appropriate professional body, which accredits them as qualifi ed practitioners. This means they must first finish medical school and then qualify as surgeons, physicians, ophthalmologists or psychiatrists.
This postgraduate training is arduous and expensive, and practitioners naturally expect a return on their investment of time, energy and money-the average medical graduate is left with tens of thousands of dollars in university fees.
Much is made of the top professionals who make millions, but the average GP is running a practice that gives him a barely adequate return on his investment in professional development. Indeed, many GPs complain they are virtually government employees relying on Medicare to pay their bills, but the "virtually" is important. They remain independent professionals who succeed or fail according to the service they provide.
The recent moves to widen the scope of nurse practitioners concern many GPs. While nurse practitioners may have a role in isolated areas, a nurse is not a substitute for a general practitioner, who has years of undergraduate and postgraduate training in family medicine. Expanding the role of nurse practitioners may simply be an axe to wield again the ancient enemy, the family GP. Many nurses have specialist training, which makes them indispensable in the medical system; but a nurse is not a substitute for professionally-trained general practitioners with years' more education behind them.
Minister Roxon's move to cut Medicare payments for cataract surgery again fl ies in the face of reality. On the face of it, it may seem plausible-better technology equals cheaper prices. If the Fred Hollows Foundation can do cataract surgery for $25, why can't an Australian ophthalmologist? The reason is that an Australian eye-doctor is running a practice. He has to pay a receptionist, an accountant, rent for his rooms and so on-in other words, he has fixed costs, which means the money goes into a lot of pockets apart from his own. In fact, he can't absorb the cost cuts that the government is asking him to accept.
From News Weekly, November 28, 2009

(1/3 of this passage's questions were captured_correct -- only those are shown below)

**Question 1:** What did the Health Minister propose?

**Gold answer:** to allocate the patients to doctors on a list basis

**Predicted answer:** that patients be allocated to doctors on a list basis (f1=0.70, conf=0.853)

**Entropy:** sent_entropy_norm=0.812, tok_entropy_norm=0.715, combined=0.763 (sent_entropy=2.614, tok_entropy=4.443, num_sentences=25)

**Answer-sentence attention:** share=0.166, rank=1/25 -- how much attention mass landed on the sentence containing the model's own predicted answer, and its rank among all sentences (1 = most-attended sentence IS the answer sentence).

**Probe question:** What is the main topic of the passage?

**Probe predicted answer:** Health Minister Nicola Roxon's latest proposal that patients be allocated to doctors on a list basis is straight out of the playbook of Britain's National Health Service.
Let's think about this from the patient's point of view (conf=0.008)

**Probe entropy:** sent_entropy_norm=0.794, tok_entropy_norm=0.609, combined=0.701

### Passage 4

**Passage:**

> The English policeman has several nicknames but the most frequently used are "copper" and "bobby". The first name comes from the verb "to cop " (which is also slang ) , meaning " to take " or " to capture ", and the second comes from the first name of Sir Robert Peel, the nineteenth-century politician, who was the founder of the police force as we know it today. An early nickname for the policeman was "peeler", but this one has died out.
Whatever we may call them, the general opinion of the police seems to be a favorable one; except, of course, among the criminal part of the community where the police are given more derogatory nicknames which originated in America, such as "fuzz" or "pig". Visitors to England seem nearly always to be very impressed by the English police. It has, in fact, become a standing joke that the visitor to Britain, when asked for his views of the country, will always say, at some point or other, "I think your policemen are wonderful. "
Well, the British bobby may not always be wonderful but he is usually a very friendly and helpful sort of character. A music-hall song of some years ago was called "If You Want To Know The Time, Ask A Policeman". Nowadays, most people own watches but they still seem to find plenty of other questions to ask the policeman. In London, the policemen spend so much of their time directing visitors about the city that one wonders how they ever find time to do anything else!
Two things are immediately noticeable to the stranger when he sees an English policeman for the first time. The first is that he does not carry a pistol and the second is that he wears a very distinctive type of headgear, the policeman's helmet. His helmet, together with his height, enable an English policeman to be seen from a considerable distance, a fact that is not without its usefulness. From time to time it is suggested that the policeman should be given a pistol and that his helmet should be taken from him, but both these suggestions are resisted by the majority of the public and the police themselves.

(1/4 of this passage's questions were captured_correct -- only those are shown below)

**Question 1:** Nowadays British people call the policeman _ .

**Gold answer:** bobby

**Predicted answer:** bobby (f1=1.00, conf=0.847)

**Entropy:** sent_entropy_norm=0.924, tok_entropy_norm=0.668, combined=0.796 (sent_entropy=2.438, tok_entropy=4.068, num_sentences=14)

**Answer-sentence attention:** share=0.047, rank=9/14 -- how much attention mass landed on the sentence containing the model's own predicted answer, and its rank among all sentences (1 = most-attended sentence IS the answer sentence).

**Probe question:** What is the main topic of the passage?

**Probe predicted answer:** criminal part of the community (conf=0.004)

**Probe entropy:** sent_entropy_norm=0.901, tok_entropy_norm=0.635, combined=0.768

### Passage 5

**Passage:**

> As supplier of most of the food we eat and of raw materials for many industrial processes, agriculture is clearly an important area of the economy. But the industrial performance of agriculture is even more important than this. For in nations where the productivity of farmers is low, most of the working population is needed to raise food and few people are available for production of investment goods or for other activities required for economic growth. Indeed, one of the factors related most closely to the per capital income  of a nation is the fraction of its population engaged in farming. In the poorest nations of the world more than half of the population lives on farms. This compares sharply with less than 10 per cent in Western Europe and less than 4 per cent in the United States.
In short, the course of economic development in general depends in a fundamental way on the performance of farmers. This performance in turn, depends on how agriculture is organized and on the economic environment, or market structure, within which it function. In the following pages the performance of American agriculture is examined. It is appropriate to begin with a conversation of its market structure.

(2/5 of this passage's questions were captured_correct -- only those are shown below)

**Question 1:** What is most important to agriculture is ________.

**Gold answer:** its industrial performance

**Predicted answer:** industrial performance (f1=0.80, conf=0.621)

**Entropy:** sent_entropy_norm=0.955, tok_entropy_norm=0.701, combined=0.828 (sent_entropy=2.198, tok_entropy=3.789, num_sentences=10)

**Answer-sentence attention:** share=0.159, rank=1/10 -- how much attention mass landed on the sentence containing the model's own predicted answer, and its rank among all sentences (1 = most-attended sentence IS the answer sentence).

**Question 2:** The performance of farmers essentially determines ________.

**Gold answer:** the general development of economy

**Predicted answer:** the course of economic development (f1=0.60, conf=0.446)

**Entropy:** sent_entropy_norm=0.924, tok_entropy_norm=0.668, combined=0.796 (sent_entropy=2.128, tok_entropy=3.610, num_sentences=10)

**Answer-sentence attention:** share=0.211, rank=1/10 -- how much attention mass landed on the sentence containing the model's own predicted answer, and its rank among all sentences (1 = most-attended sentence IS the answer sentence).

**Probe question:** What is the main topic of the passage?

**Probe predicted answer:** its market structure (conf=0.729)

**Probe entropy:** sent_entropy_norm=0.972, tok_entropy_norm=0.674, combined=0.823

### Passage 6

**Passage:**

> The biggest safety threat facing airlines today may not be a terrorist with a gun, but the man with the portable computer in business class. In the last 15 years, pilots have reported well over 100 incidents that could have been caused by electromagnetic interference. The source of this interference remains unconfirmed, but increasingly, experts are pointing the blame at portable electronic device such as portable computers, radio and cassette players and mobile telephones.
RTCA, an organization which advises the aviation  industry, has recommended that all airlines ban  such devices from being used during "critical" stages of flight, particularly take-off and landing. Some experts have gone further, calling for a total ban during all flights. Currently, rules on using these devices are left up to individual airlines. And although some airlines prohibit passengers from using such equipment during take-off and landing, most are reluctant to enforce a total ban, given that many passengers want to work during flights.
The difficulty is predicting how electromagnetic fields might affect an aircraft's computers. Experts know that portable device emit radiation which affects those wavelengths which aircraft use for navigation and communication. But, because they have not been able to reproduce these effects in a laboratory, they have no way of knowing whether the interference might be dangerous or not.
The fact that aircraft may be vulnerable  to interference raises the risk that terrorists may use radio systems in order to damage navigation equipment. As worrying, though, is the passenger who can't hear the instructions to turn off his radio because the music's too loud.

(1/5 of this passage's questions were captured_correct -- only those are shown below)

**Question 1:** Why is it difficult to predict the possible effects of electromagnetic fields on an airplane's computers?

**Gold answer:** Because research scientists have not been able to produce the same effects in labs.

**Predicted answer:** because they have not been able to reproduce these effects in a laboratory (f1=0.59, conf=0.579)

**Entropy:** sent_entropy_norm=0.932, tok_entropy_norm=0.714, combined=0.823 (sent_entropy=2.317, tok_entropy=4.092, num_sentences=12)

**Answer-sentence attention:** share=0.152, rank=1/12 -- how much attention mass landed on the sentence containing the model's own predicted answer, and its rank among all sentences (1 = most-attended sentence IS the answer sentence).

**Probe question:** What is the main topic of the passage?

**Probe predicted answer:** the man with the portable computer in business class (conf=0.652)

**Probe entropy:** sent_entropy_norm=0.904, tok_entropy_norm=0.690, combined=0.797

Avg over 7 captured=Y question(s): sent_entropy_norm=0.858, tok_entropy_norm=0.657, combined=0.757 (probe combined avg=0.742); answer_sentence_attention_share=0.143, answer_sentence_rank=3.14 (n_located=7)

## OneStopQA / EASY

6/6 captured (tried 11 candidates)

### Passage 1

**Passage:**

> After two years of successful ads with cute animals – a bear and hare, then a penguin – this time, the story is about a young girl, Lily, who sees an old man living in a small wooden house on the moon through her telescope. The girl first tries to send him a letter and a note via bow and arrow. Then, she floats him a present of a telescope tied to balloons. This finally allows them to make contact. The ad’s message is: “Show someone they’re loved this Christmas.” This is similar to Age UK’s campaign: “No one should have no one at Christmas.” Profits from three products – a mug, gift tag and card – will go to the charity. Rachel Swift, head of marketing at John Lewis, said that people talk about charities at Christmas and the ad makes you think about someone who lives on your street that might not see anybody.

(1/3 of this passage's questions were captured_correct -- only those are shown below)

**Question 1:** Who does Rachel Swift work for?

**Gold answer:** John Lewis

**Predicted answer:** John Lewis (f1=1.00, conf=0.998)

**Entropy:** sent_entropy_norm=0.859, tok_entropy_norm=0.701, combined=0.780 (sent_entropy=1.786, tok_entropy=3.693, num_sentences=8)

**Answer-sentence attention:** share=0.283, rank=1/8 -- how much attention mass landed on the sentence containing the model's own predicted answer, and its rank among all sentences (1 = most-attended sentence IS the answer sentence).

**Probe question:** What is the main topic of the passage?

**Probe predicted answer:** a young girl, Lily, who sees an old man living in a small wooden house on the moon through her telescope (conf=0.267)

**Probe entropy:** sent_entropy_norm=0.757, tok_entropy_norm=0.589, combined=0.673

### Passage 2

**Passage:**

> Benjamin Carle is 96.9% made in France, even his underpants and socks. Six Ikea forks, a Chinese guitar and some wall paint stopped him being called 100% French, but nobody is perfect. Carle, 26, decided, in 2013, to see if it was possible to live using only French-made products for ten months as part of a television documentary. He got the idea after the Minister for Economic Renewal, Arnaud Montebourg, asked the French people to buy French products. For the experiment, Carle had to give up his smartphone, television, refrigerator (all made in China); his glasses (Italian); his morning coffee (Guatemalan) and his favourite David Bowie music (British). It is lucky that his girlfriend, Anaïs, and cat, Loon, are both French, so he didn’t have to give them up.

(1/3 of this passage's questions were captured_correct -- only those are shown below)

**Question 1:** What did Arnaud Montebourgh do?

**Gold answer:** He asked people to buy products made in France

**Predicted answer:** asked the French people to buy French products (f1=0.59, conf=0.870)

**Entropy:** sent_entropy_norm=0.870, tok_entropy_norm=0.722, combined=0.796 (sent_entropy=1.694, tok_entropy=3.761, num_sentences=7)

**Answer-sentence attention:** share=0.235, rank=1/7 -- how much attention mass landed on the sentence containing the model's own predicted answer, and its rank among all sentences (1 = most-attended sentence IS the answer sentence).

**Probe question:** What is the main topic of the passage?

**Probe predicted answer:** nobody is perfect. Carle, 26, decided, in 2013, to see if it was possible to live using only French-made products for ten months as part of a television documentary (conf=0.138)

**Probe entropy:** sent_entropy_norm=0.840, tok_entropy_norm=0.600, combined=0.720

### Passage 3

**Passage:**

> The department store John Lewis has a 2015 Christmas advertisement. The ad shows a lonely old man who lives on the moon. The ad, which for many people shows that the Christmas shopping season has begun, aims to raise hundreds of thousands of pounds for the charity Age UK. John Lewis will also encourage staff and customers to care for elderly people who might be alone over the holiday. The department store has spent £7 million on a campaign that includes the TV ad, a smartphone game and merchandise, including glow-in-the-dark pyjamas. It will also build areas that look like the surface of the moon in 11 of its stores.

(1/3 of this passage's questions were captured_correct -- only those are shown below)

**Question 1:** Where does the lonely old man appear in John Lewis’s advertisement?

**Gold answer:** On the moon

**Predicted answer:** on the moon (f1=1.00, conf=0.572)

**Entropy:** sent_entropy_norm=0.883, tok_entropy_norm=0.704, combined=0.794 (sent_entropy=1.583, tok_entropy=3.419, num_sentences=6)

**Answer-sentence attention:** share=0.026, rank=6/6 -- how much attention mass landed on the sentence containing the model's own predicted answer, and its rank among all sentences (1 = most-attended sentence IS the answer sentence).

**Probe question:** What is the main topic of the passage?

**Probe predicted answer:** Christmas advertisement. The ad shows a lonely old man who lives on the moon. The ad, which for many people shows that the Christmas shopping season has begun, aims to raise hundreds of thousands of pounds for the charity Age UK (conf=0.081)

**Probe entropy:** sent_entropy_norm=0.847, tok_entropy_norm=0.520, combined=0.683

### Passage 4

**Passage:**

> Autism is a disorder that one in 100 people have. It affects people in different ways, but causes difficulties in social interaction and communication. So far, there is no effective treatment for the social problems that autism causes. Researchers at Yale have studied the brain chemical oxytocin. They say it is a possible treatment for the social problems caused by autism because it plays an important role in bonding and trust. But not all results are positive: one recent study found no significant benefit for young people who took the chemical for several days. But Pelphrey said oxytocin might help the brain learn from social interactions; it would work best when used together with therapies that encourage people with autism to interact more socially, he said.

(1/3 of this passage's questions were captured_correct -- only those are shown below)

**Question 1:** How were the social aspects of autism effectively treated before the time this article was written?

**Gold answer:** There was no effective treatment available

**Predicted answer:** there is no effective treatment (f1=0.73, conf=0.385)

**Entropy:** sent_entropy_norm=0.893, tok_entropy_norm=0.683, combined=0.788 (sent_entropy=1.737, tok_entropy=3.400, num_sentences=7)

**Answer-sentence attention:** share=0.031, rank=6/7 -- how much attention mass landed on the sentence containing the model's own predicted answer, and its rank among all sentences (1 = most-attended sentence IS the answer sentence).

**Probe question:** What is the main topic of the passage?

**Probe predicted answer:** Autism (conf=0.871)

**Probe entropy:** sent_entropy_norm=0.828, tok_entropy_norm=0.541, combined=0.684

### Passage 5

**Passage:**

> Scientists must find new names for the elements but, also, they must suggest two-letter symbols for the elements. When IUPAC has received the researchers’ suggestions, they will tell the public so that people can comment on the names. That allows scientists and others to find any problems with the names. In 1996, someone suggested the symbol Cp for copernicium, or element 112, but it was changed to Cn, when scientists complained that Cp was already the symbol for another substance.

(1/3 of this passage's questions were captured_correct -- only those are shown below)

**Question 1:** Why was the suggestion to use symbol Cp for copernicium not accepted?

**Gold answer:** The symbol was already being used for a different substance

**Predicted answer:** Cp was already the symbol for another substance (f1=0.67, conf=0.789)

**Entropy:** sent_entropy_norm=0.883, tok_entropy_norm=0.670, combined=0.776 (sent_entropy=1.224, tok_entropy=3.104, num_sentences=4)

**Answer-sentence attention:** share=0.439, rank=1/4 -- how much attention mass landed on the sentence containing the model's own predicted answer, and its rank among all sentences (1 = most-attended sentence IS the answer sentence).

**Probe question:** What is the main topic of the passage?

**Probe predicted answer:** Scientists must find new names for the elements but, also, they must suggest two-letter symbols for the elements. When IUPAC has received the researchers’ suggestions, they will tell the public so that people can comment on the names. That allows scientists and others to find any problems with the names (conf=0.104)

**Probe entropy:** sent_entropy_norm=0.851, tok_entropy_norm=0.481, combined=0.666

### Passage 6

**Passage:**

> Most of our customers are “baby boomers who want to have the cycling experience they had as a kid,” says Pedego’s Don DiCostanza. “The main reason they stopped riding bikes was because of hills.” Pedego has opened nearly 60 stores in the US. ElectroBike has 30 stores in Mexico. It opened its first American store in Venice Beach, California in 2014 and hopes to have 25 US stores in a year. CEO Craig Anderson says: “We want to help reduce traffic, help reduce our carbon footprint and encourage a healthy lifestyle.” He tells customers: “Ride this bike once and try not to smile.” Startups like Pedego and ElectroBike will have to compete with big companies like Trek, Currie, and Accell - the market leader in e-bikes in Europe. Accell owns the Raleigh brand, as well as Haibike, an award-winning German electric bike.

(1/3 of this passage's questions were captured_correct -- only those are shown below)

**Question 1:** What does ElectroBike hope to accomplish within a year of opening its store in Venice Beach?

**Gold answer:** Have 25 stores in the US

**Predicted answer:** hopes to have 25 US stores in a year (f1=0.67, conf=0.128)

**Entropy:** sent_entropy_norm=0.919, tok_entropy_norm=0.698, combined=0.808 (sent_entropy=2.019, tok_entropy=3.711, num_sentences=9)

**Answer-sentence attention:** share=0.077, rank=7/9 -- how much attention mass landed on the sentence containing the model's own predicted answer, and its rank among all sentences (1 = most-attended sentence IS the answer sentence).

**Probe question:** What is the main topic of the passage?

**Probe predicted answer:** reduce traffic (conf=0.312)

**Probe entropy:** sent_entropy_norm=0.926, tok_entropy_norm=0.624, combined=0.775

Avg over 6 captured=Y question(s): sent_entropy_norm=0.884, tok_entropy_norm=0.696, combined=0.790 (probe combined avg=0.700); answer_sentence_attention_share=0.182, answer_sentence_rank=3.67 (n_located=6)

## OneStopQA / MEDIUM

6/6 captured (tried 9 candidates)

### Passage 1

**Passage:**

> The potential for solar power from the desert has been known for decades. In the days after the Chernobyl nuclear accident in 1986, the German particle physicist Gerhard Knies calculated that the world’s deserts receive enough energy in a few hours to provide power for all the people in the world for a whole year. But the challenge is to capture that energy and take it to where it is needed. Experts say that solar energy will make up a third of Morocco’s renewable energy supply by 2020. Wind and hydro will make up the other two-thirds. “We are very proud of this project,” Morocco’s environment minister, Hakima el-Haite said. “I think it is the most important solar plant in the world.”

(1/3 of this passage's questions were captured_correct -- only those are shown below)

**Question 1:** By 2020, two-thirds of Morocco’s renewable energy will be ...

**Gold answer:** Hydro and wind energy

**Predicted answer:** Wind and hydro (f1=0.86, conf=0.648)

**Entropy:** sent_entropy_norm=0.792, tok_entropy_norm=0.536, combined=0.664 (sent_entropy=1.647, tok_entropy=2.715, num_sentences=8)

**Answer-sentence attention:** share=0.241, rank=1/8 -- how much attention mass landed on the sentence containing the model's own predicted answer, and its rank among all sentences (1 = most-attended sentence IS the answer sentence).

**Probe question:** What is the main topic of the passage?

**Probe predicted answer:** the challenge is to capture that energy and take it to where it is needed (conf=0.315)

**Probe entropy:** sent_entropy_norm=0.793, tok_entropy_norm=0.570, combined=0.682

### Passage 2

**Passage:**

> Vienna is the world’s best city to live in, Baghdad is the worst and London, Paris and New York do not even enter the top 35, according to international research into quality of life. German-speaking cities dominate the rankings in the 18th Mercer Quality of Life study, with Vienna joined by Zurich, Munich, Dusseldorf and Frankfurt in the top seven. Paris has dropped down the table – it has fallen ten places to 37th, just ahead of London at 39th, mostly because of the terrorist attacks on the city. The study examined social and economic conditions, health, education, housing and the environment. It is used by big companies to decide where they should open offices and factories and how much they should pay staff.

(1/3 of this passage's questions were captured_correct -- only those are shown below)

**Question 1:** What is the ranking of Paris in the 18th Mercer Quality of Life study?

**Gold answer:** 37th

**Predicted answer:** 37th (f1=1.00, conf=0.981)

**Entropy:** sent_entropy_norm=0.858, tok_entropy_norm=0.574, combined=0.716 (sent_entropy=1.380, tok_entropy=2.889, num_sentences=5)

**Answer-sentence attention:** share=0.344, rank=1/5 -- how much attention mass landed on the sentence containing the model's own predicted answer, and its rank among all sentences (1 = most-attended sentence IS the answer sentence).

**Probe question:** What is the main topic of the passage?

**Probe predicted answer:** The study examined social and economic conditions, health, education, housing and the environment (conf=0.228)

**Probe entropy:** sent_entropy_norm=0.810, tok_entropy_norm=0.558, combined=0.684

### Passage 3

**Passage:**

> According to American researchers, a nasal spray containing the ‘Love hormone’ oxytocin could help children with autism behave more normally in social situations. Scans of autistic children showed that a single dose of the chemical improved brain responses to facial expressions. This is something that could make social interactions feel more natural and rewarding for them. The researchers said oxytocin might increase the success of behavioral therapies that are already used to help people with autism learn to cope with social situations “Over time, what you would expect to see is more normal social responding, being more interested in interacting with other people, more eye contact and more conversation,” said Kevin Pelphrey, of Yale University.

(1/3 of this passage's questions were captured_correct -- only those are shown below)

**Question 1:** What difference did oxytocin make to the brains of autistic children, as shown in brain scans?

**Gold answer:** Improved responses to faces

**Predicted answer:** improved brain responses to facial expressions (f1=0.60, conf=0.894)

**Entropy:** sent_entropy_norm=0.822, tok_entropy_norm=0.620, combined=0.721 (sent_entropy=1.139, tok_entropy=3.064, num_sentences=4)

**Answer-sentence attention:** share=0.286, rank=2/4 -- how much attention mass landed on the sentence containing the model's own predicted answer, and its rank among all sentences (1 = most-attended sentence IS the answer sentence).

**Probe question:** What is the main topic of the passage?

**Probe predicted answer:** oxytocin could help children with autism (conf=0.066)

**Probe entropy:** sent_entropy_norm=0.799, tok_entropy_norm=0.510, combined=0.655

### Passage 4

**Passage:**

> Part of the reason for this is that air travel is dangerous so standards are much higher. “If you fly commercial airlines, they often say, ‘Oh, a small component has failed – we have to go back to the gate,’” Singh said. “And that’s an established industry with 60 years of legacy! I hate to think that a drone might come down on a busy road.” Part of the solution, Singh said, is planning for every situation: “If things fail, the vehicle has to do something reasonable.”

(1/3 of this passage's questions were captured_correct -- only those are shown below)

**Question 1:** According to Singh, why do passenger airplanes often return to the gate?

**Gold answer:** A small part has failed

**Predicted answer:** a small component has failed (f1=0.80, conf=0.836)

**Entropy:** sent_entropy_norm=0.839, tok_entropy_norm=0.713, combined=0.776 (sent_entropy=1.504, tok_entropy=3.405, num_sentences=6)

**Answer-sentence attention:** share=0.236, rank=2/6 -- how much attention mass landed on the sentence containing the model's own predicted answer, and its rank among all sentences (1 = most-attended sentence IS the answer sentence).

**Probe question:** What is the main topic of the passage?

**Probe predicted answer:** planning for every situation (conf=0.342)

**Probe entropy:** sent_entropy_norm=0.798, tok_entropy_norm=0.513, combined=0.656

### Passage 5

**Passage:**

> Do you want your child to be good at sports, play for the school team and, maybe one day, even compete in international competitions? Well, try to make sure that your future Olympian or World Cup winner is born in November or October. A study has found that school pupils born in those months are fitter than everyone else in their class. November- and October-born children were fitter, stronger and more powerful than those born in the other ten months of the year, especially those whose birthdays were in April or June. Dr. Gavin Sandercock of Essex University found in his study that autumn-born children had “a clear physical advantage” over their classmates.

(1/3 of this passage's questions were captured_correct -- only those are shown below)

**Question 1:** Who does the article suggest to be the weakest in comparison with children born in October and November?

**Gold answer:** Children born in April and June

**Predicted answer:** November- and October-born children were fitter, stronger and more powerful than those born in the other ten months of the year, especially those whose birthdays were in April or June (f1=0.33, conf=0.073)

**Entropy:** sent_entropy_norm=0.714, tok_entropy_norm=0.592, combined=0.653 (sent_entropy=1.279, tok_entropy=2.928, num_sentences=6)

**Answer-sentence attention:** share=0.404, rank=1/6 -- how much attention mass landed on the sentence containing the model's own predicted answer, and its rank among all sentences (1 = most-attended sentence IS the answer sentence).

**Probe question:** What is the main topic of the passage?

**Probe predicted answer:** try to make sure that your future Olympian or World Cup winner is born in November or October (conf=0.313)

**Probe entropy:** sent_entropy_norm=0.598, tok_entropy_norm=0.460, combined=0.529

### Passage 6

**Passage:**

> The ice-cream shop is in a documentary by film-makers Rob and Lisa Fruchtman. Sweet Dreams, which tells the story of how the women have made a promising post-genocide future, also includes the female drummers. The film has been shown in more than a dozen countries, including the US, UK and several African states. “We feel the film is about resilience, hope, bravery, resourcefulness and the ability to change the course of your own life,” says Lisa Fruchtman.

(1/3 of this passage's questions were captured_correct -- only those are shown below)

**Question 1:** What is the name of Rob and Lisa Fruchtman’s film?

**Gold answer:** Sweet Dreams

**Predicted answer:** Sweet Dreams (f1=1.00, conf=0.998)

**Entropy:** sent_entropy_norm=0.953, tok_entropy_norm=0.600, combined=0.777 (sent_entropy=1.322, tok_entropy=2.808, num_sentences=4)

**Answer-sentence attention:** share=0.270, rank=3/4 -- how much attention mass landed on the sentence containing the model's own predicted answer, and its rank among all sentences (1 = most-attended sentence IS the answer sentence).

**Probe question:** What is the main topic of the passage?

**Probe predicted answer:** how the women have made a promising post-genocide future (conf=0.431)

**Probe entropy:** sent_entropy_norm=0.888, tok_entropy_norm=0.519, combined=0.703

Avg over 6 captured=Y question(s): sent_entropy_norm=0.830, tok_entropy_norm=0.606, combined=0.718 (probe combined avg=0.651); answer_sentence_attention_share=0.297, answer_sentence_rank=1.67 (n_located=6)

## OneStopQA / HARD

6/6 captured (tried 13 candidates)

### Passage 1

**Passage:**

> South American Indians have chewed coca leaves for centuries. The leaves reputedly provide energy and are said to have medicinal qualities. Supporters of Bolivia’s position praised it for doing the right thing by defending the rights of indigenous people. “The Bolivian move is inspirational and groundbreaking,” said Danny Kushlick, Head of External Affairs at the Transform Drug Policy Foundation, which promotes drug liberalization. “It shows that any country that has had enough of the war on drugs can change the terms of its engagement with the UN conventions.”

(1/3 of this passage's questions were captured_correct -- only those are shown below)

**Question 1:** Who is Danny Kushlick?

**Gold answer:** A member of the Transform Drug Policy Foundation

**Predicted answer:** Head of External Affairs at the Transform Drug Policy Foundation (f1=0.67, conf=0.969)

**Entropy:** sent_entropy_norm=0.679, tok_entropy_norm=0.574, combined=0.626 (sent_entropy=1.217, tok_entropy=2.726, num_sentences=6)

**Answer-sentence attention:** share=0.372, rank=1/6 -- how much attention mass landed on the sentence containing the model's own predicted answer, and its rank among all sentences (1 = most-attended sentence IS the answer sentence).

**Probe question:** What is the main topic of the passage?

**Probe predicted answer:** war on drugs (conf=0.709)

**Probe entropy:** sent_entropy_norm=0.642, tok_entropy_norm=0.474, combined=0.558

### Passage 2

**Passage:**

> n octopus has made a brazen escape from the National Aquarium in New Zealand by breaking out of its tank, slithering down a 50-meter drainpipe and disappearing into the sea. In scenes reminiscent of Finding Nemo, Inky – a common New Zealand octopus – made his dash for freedom after the lid of his tank was accidentally left slightly ajar. Staff believe that in the middle of the night, while the aquarium was deserted, Inky clambered to the top of his glass enclosure, down the side of the tank and traveled across the floor of the aquarium. Rob Yarrell, national manager of the National Aquarium of New Zealand in Napier, said: “Octopuses are famous escape artists. I don’t think he was unhappy with us, or lonely, as octopuses are solitary creatures. But, he is such a curious boy. He would want to know what’s happening on the outside. That’s just his personality.”

(1/3 of this passage's questions were captured_correct -- only those are shown below)

**Question 1:** How does Yarrell describe Inky’s personality?

**Gold answer:** Curious

**Predicted answer:** curious boy (f1=0.67, conf=0.530)

**Entropy:** sent_entropy_norm=0.895, tok_entropy_norm=0.729, combined=0.812 (sent_entropy=1.966, tok_entropy=3.878, num_sentences=9)

**Answer-sentence attention:** share=0.009, rank=8/9 -- how much attention mass landed on the sentence containing the model's own predicted answer, and its rank among all sentences (1 = most-attended sentence IS the answer sentence).

**Probe question:** What is the main topic of the passage?

**Probe predicted answer:** outside (conf=0.000)

**Probe entropy:** sent_entropy_norm=0.894, tok_entropy_norm=0.597, combined=0.746

### Passage 3

**Passage:**

> From all across Rwanda, and even parts of neighboring Burundi, people flock to the southern town of Butare to a little shop called Inzozi Nziza (Sweet Dreams). They come for a taste of the unknown, something most have never tasted – the sweet, cold, velvety embrace of ice cream. Here, at the central African country’s first ice-cream parlor, customers can buy scoops in sweet cream, passion fruit, strawberry and pineapple flavors. Toppings include fresh fruit, honey, chocolate chips and granola. Black tea and coffee are also on sale.

(1/3 of this passage's questions were captured_correct -- only those are shown below)

**Question 1:** Where is the ice cream shop located?

**Gold answer:** Butare

**Predicted answer:** the southern town of Butare (f1=0.33, conf=0.603)

**Entropy:** sent_entropy_norm=0.869, tok_entropy_norm=0.612, combined=0.741 (sent_entropy=1.399, tok_entropy=2.951, num_sentences=5)

**Answer-sentence attention:** share=0.088, rank=4/5 -- how much attention mass landed on the sentence containing the model's own predicted answer, and its rank among all sentences (1 = most-attended sentence IS the answer sentence).

**Probe question:** What is the main topic of the passage?

**Probe predicted answer:** a taste of the unknown (conf=0.283)

**Probe entropy:** sent_entropy_norm=0.847, tok_entropy_norm=0.525, combined=0.686

### Passage 4

**Passage:**

> But it isn’t just students who would benefit from a later start. Kelley says the working day should be more forgiving of our natural rhythms. Describing the average sleep loss per night for different age groups, he says: “Between 14 and 24, it’s more than two hours. For people aged between 24 and about 30 or 35, it’s about an hour and a half. That can continue up until you’re about 55 when it’s in balance again. The 10-year-old and 55-year-old wake and sleep naturally at the same time.”

(1/3 of this passage's questions were captured_correct -- only those are shown below)

**Question 1:** What does Kelley say about the relationship between working hours and natural rhythms?

**Gold answer:** The working day should be adjusted to our natural rhythms

**Predicted answer:** the working day should be more forgiving (f1=0.59, conf=0.588)

**Entropy:** sent_entropy_norm=0.733, tok_entropy_norm=0.543, combined=0.638 (sent_entropy=1.425, tok_entropy=2.629, num_sentences=7)

**Answer-sentence attention:** share=0.011, rank=6/7 -- how much attention mass landed on the sentence containing the model's own predicted answer, and its rank among all sentences (1 = most-attended sentence IS the answer sentence).

**Probe question:** What is the main topic of the passage?

**Probe predicted answer:** the working day should be more forgiving of our natural rhythms (conf=0.200)

**Probe entropy:** sent_entropy_norm=0.749, tok_entropy_norm=0.522, combined=0.635

### Passage 5

**Passage:**

> Huber said about Amazon: “I have heard them say that many packages are lightweight – a drone can carry a kilogram for 15 minutes. If you have a vehicle that can go into a neighborhood, it can deliver from that base. You need a 15-minute distance and typical off-the-shelf drones have about that distance.” It’s one way, he said, of making sure the surrounding population is relatively safe. “The larger the distance the drone travels, the more dangerous it becomes.” Of course, safety remains a major concern – Singh points out that, for a commercial aircraft to be considered skyworthy, it has to prove a rate of one serious failure every one million hours. Drones, he said, are “one or two orders of magnitude away” from that benchmark. “The Reaper drone has one failure in 10,000 hours,” Singh said. An oil leak, by the way, doesn’t count as catastrophic failure – something has to fall out of the sky.

(2/3 of this passage's questions were captured_correct -- only those are shown below)

**Question 1:** What happens as you increase the distance a drone travels to deliver a package?

**Gold answer:** The drone becomes more dangerous to people

**Predicted answer:** the more dangerous (f1=0.60, conf=0.405)

**Entropy:** sent_entropy_norm=0.917, tok_entropy_norm=0.737, combined=0.827 (sent_entropy=2.015, tok_entropy=3.960, num_sentences=9)

**Answer-sentence attention:** share=0.133, rank=6/9 -- how much attention mass landed on the sentence containing the model's own predicted answer, and its rank among all sentences (1 = most-attended sentence IS the answer sentence).

**Question 2:** What determines if a passenger plane is allowed to operate?

**Gold answer:** The number of serious failures every one million hours

**Predicted answer:** one serious failure every one million hours (f1=0.63, conf=0.406)

**Entropy:** sent_entropy_norm=0.948, tok_entropy_norm=0.716, combined=0.832 (sent_entropy=2.084, tok_entropy=3.847, num_sentences=9)

**Answer-sentence attention:** share=0.182, rank=1/9 -- how much attention mass landed on the sentence containing the model's own predicted answer, and its rank among all sentences (1 = most-attended sentence IS the answer sentence).

**Probe question:** What is the main topic of the passage?

**Probe predicted answer:** safety (conf=0.558)

**Probe entropy:** sent_entropy_norm=0.924, tok_entropy_norm=0.620, combined=0.772

### Passage 6

**Passage:**

> The loans Duran swindled from banks were his way of regulating and denouncing this situation, he said. He started slowly. “I filled out a few credit applications with my real details. They denied me, but I just wanted to get a feel for what they were asking for.” From there, the former table-tennis coach began to weave an intricate web of accounts, payments and transfers. “I was learning constantly.” By the summer of 2007, he had discovered how to make the system work, applying for loans under the name of a false television production company. “Then, I managed to get a lot.” €492,000, to be exact.

(1/3 of this passage's questions were captured_correct -- only those are shown below)

**Question 1:** How much money did Duran manage to take out in loans?

**Gold answer:** €492,000 in total

**Predicted answer:** €492,000 (f1=0.50, conf=0.997)

**Entropy:** sent_entropy_norm=0.925, tok_entropy_norm=0.682, combined=0.804 (sent_entropy=2.032, tok_entropy=3.391, num_sentences=9)

**Answer-sentence attention:** share=0.171, rank=2/9 -- how much attention mass landed on the sentence containing the model's own predicted answer, and its rank among all sentences (1 = most-attended sentence IS the answer sentence).

**Probe question:** What is the main topic of the passage?

**Probe predicted answer:** learning (conf=0.059)

**Probe entropy:** sent_entropy_norm=0.926, tok_entropy_norm=0.681, combined=0.803

Avg over 7 captured=Y question(s): sent_entropy_norm=0.852, tok_entropy_norm=0.656, combined=0.754 (probe combined avg=0.700); answer_sentence_attention_share=0.138, answer_sentence_rank=4.00 (n_located=7)

## SQuAD / N/A

6/6 captured (tried 6 candidates)

### Passage 1

**Passage:**

> Although sizable Orthodox Jewish communities are located throughout the United States, many American Orthodox Jews live in New York State, particularly in the New York City Metropolitan Area. Two of the main Orthodox communities in the United States are located in New York City and Rockland County. In New York City, the neighborhoods of Borough Park, Midwood, Williamsburg, and Crown Heights, located in the borough of Brooklyn, have particularly large Orthodox communities. The most rapidly growing community of American Orthodox Jews is located in Rockland County and the Hudson Valley of New York, including the communities of Monsey, Monroe, New Square, and Kiryas Joel. There are also sizable and rapidly growing Orthodox communities throughout New Jersey, particularly in Lakewood, Teaneck, Englewood, Passaic, and Fair Lawn.

(4/4 of this passage's questions were captured_correct -- only those are shown below)

**Question 1:** Borough Park, Midwood, Williamsburg and Crown heights have particularly large communities of what?

**Gold answer:** American Orthodox Jews

**Predicted answer:** American Orthodox Jews (f1=1.00, conf=0.432)

**Entropy:** sent_entropy_norm=0.990, tok_entropy_norm=0.561, combined=0.775 (sent_entropy=1.593, tok_entropy=2.851, num_sentences=5)

**Answer-sentence attention:** share=0.176, rank=4/5 -- how much attention mass landed on the sentence containing the model's own predicted answer, and its rank among all sentences (1 = most-attended sentence IS the answer sentence).

**Question 2:** Where is a sizeable and rapidly growing Orthodox community currently located besides New York State?

**Gold answer:** New Jersey

**Predicted answer:** New Jersey (f1=1.00, conf=0.984)

**Entropy:** sent_entropy_norm=0.911, tok_entropy_norm=0.636, combined=0.773 (sent_entropy=1.466, tok_entropy=3.231, num_sentences=5)

**Answer-sentence attention:** share=0.270, rank=2/5 -- how much attention mass landed on the sentence containing the model's own predicted answer, and its rank among all sentences (1 = most-attended sentence IS the answer sentence).

**Question 3:** Where is the most rapidly growing community of American orthodox jews located?

**Gold answer:** Rockland County

**Predicted answer:** Rockland County (f1=1.00, conf=0.360)

**Entropy:** sent_entropy_norm=0.856, tok_entropy_norm=0.523, combined=0.690 (sent_entropy=1.378, tok_entropy=2.656, num_sentences=5)

**Answer-sentence attention:** share=0.007, rank=5/5 -- how much attention mass landed on the sentence containing the model's own predicted answer, and its rank among all sentences (1 = most-attended sentence IS the answer sentence).

**Question 4:** Where do many American Orthodox Jews live?

**Gold answer:** New York State

**Predicted answer:** New York State (f1=1.00, conf=0.861)

**Entropy:** sent_entropy_norm=0.847, tok_entropy_norm=0.642, combined=0.744 (sent_entropy=1.363, tok_entropy=3.261, num_sentences=5)

**Answer-sentence attention:** share=0.071, rank=4/5 -- how much attention mass landed on the sentence containing the model's own predicted answer, and its rank among all sentences (1 = most-attended sentence IS the answer sentence).

**Probe question:** What is the main topic of the passage?

**Probe predicted answer:** Orthodox (conf=0.111)

**Probe entropy:** sent_entropy_norm=0.733, tok_entropy_norm=0.387, combined=0.560

### Passage 2

**Passage:**

> These areas, quartiers sensibles ("sensitive quarters"), are in northern and eastern Paris, namely around its Goutte d'Or and Belleville neighbourhoods. To the north of the city they are grouped mainly in the Seine-Saint-Denis department, and to a lesser extreme to the east in the Val-d'Oise department. Other difficult areas are located in the Seine valley, in Évry et Corbeil-Essonnes (Essonne), in Mureaux, Mantes-la-Jolie (Yvelines), and scattered among social housing districts created by Delouvrier's 1961 "ville nouvelle" political initiative.

(3/3 of this passage's questions were captured_correct -- only those are shown below)

**Question 1:** Where are the quartiers sensibles located?

**Gold answer:** northern and eastern Paris

**Predicted answer:** northern and eastern Paris (f1=1.00, conf=0.958)

**Entropy:** sent_entropy_norm=0.997, tok_entropy_norm=0.598, combined=0.797 (sent_entropy=1.095, tok_entropy=2.959, num_sentences=3)

**Answer-sentence attention:** share=0.328, rank=2/3 -- how much attention mass landed on the sentence containing the model's own predicted answer, and its rank among all sentences (1 = most-attended sentence IS the answer sentence).

**Question 2:** What two neighborhoods are the centers of the quartiers sensibles?

**Gold answer:** Goutte d'Or and Belleville

**Predicted answer:** Goutte d'Or and Belleville (f1=1.00, conf=0.997)

**Entropy:** sent_entropy_norm=0.992, tok_entropy_norm=0.626, combined=0.809 (sent_entropy=1.090, tok_entropy=3.100, num_sentences=3)

**Answer-sentence attention:** share=0.328, rank=2/3 -- how much attention mass landed on the sentence containing the model's own predicted answer, and its rank among all sentences (1 = most-attended sentence IS the answer sentence).

**Question 3:** Why were these neighborhoods created?

**Gold answer:** Delouvrier's 1961 "ville nouvelle" political initiative

**Predicted answer:** Delouvrier's 1961 "ville nouvelle" political initiative (f1=1.00, conf=0.518)

**Entropy:** sent_entropy_norm=0.843, tok_entropy_norm=0.603, combined=0.723 (sent_entropy=0.926, tok_entropy=2.983, num_sentences=3)

**Answer-sentence attention:** share=0.572, rank=1/3 -- how much attention mass landed on the sentence containing the model's own predicted answer, and its rank among all sentences (1 = most-attended sentence IS the answer sentence).

**Probe question:** What is the main topic of the passage?

**Probe predicted answer:** difficult (conf=0.073)

**Probe entropy:** sent_entropy_norm=0.982, tok_entropy_norm=0.460, combined=0.721

### Passage 3

**Passage:**

> When Emperor Kammu moved the capital to Heian-kyō (Kyōto), which remained the imperial capital for the next 1,000 years, he did so not only to strengthen imperial authority but also to improve his seat of government geopolitically. Nara was abandoned after only 70 years in part due to the ascendancy of Dōkyō and the encroaching secular power of the Buddhist institutions there. Kyōto had good river access to the sea and could be reached by land routes from the eastern provinces. The early Heian period (784–967) continued Nara culture; the Heian capital was patterned on the Chinese Tang capital at Chang'an, as was Nara, but on a larger scale than Nara. Kammu endeavoured to improve the Tang-style administrative system which was in use. Known as the ritsuryō, this system attempted to recreate the Tang imperium in Japan, despite the "tremendous differences in the levels of development between the two countries". Despite the decline of the Taika-Taihō reforms, imperial government was vigorous during the early Heian period. Indeed, Kammu's avoidance of drastic reform decreased the intensity of political struggles, and he became recognized as one of Japan's most forceful emperors.

(5/5 of this passage's questions were captured_correct -- only those are shown below)

**Question 1:** Heian was Japan's capital for how many years?

**Gold answer:** 1,000

**Predicted answer:** 1,000 (f1=1.00, conf=0.817)

**Entropy:** sent_entropy_norm=0.949, tok_entropy_norm=0.712, combined=0.830 (sent_entropy=1.974, tok_entropy=3.963, num_sentences=8)

**Answer-sentence attention:** share=0.212, rank=1/8 -- how much attention mass landed on the sentence containing the model's own predicted answer, and its rank among all sentences (1 = most-attended sentence IS the answer sentence).

**Question 2:** Nara was the former capital for how many years?

**Gold answer:** 70

**Predicted answer:** 70 (f1=1.00, conf=0.897)

**Entropy:** sent_entropy_norm=0.949, tok_entropy_norm=0.680, combined=0.814 (sent_entropy=1.973, tok_entropy=3.786, num_sentences=8)

**Answer-sentence attention:** share=0.050, rank=7/8 -- how much attention mass landed on the sentence containing the model's own predicted answer, and its rank among all sentences (1 = most-attended sentence IS the answer sentence).

**Question 3:** What religion was gaining popularity in Nara?

**Gold answer:** Buddhist

**Predicted answer:** Buddhist (f1=1.00, conf=0.999)

**Entropy:** sent_entropy_norm=0.967, tok_entropy_norm=0.747, combined=0.857 (sent_entropy=2.011, tok_entropy=4.157, num_sentences=8)

**Answer-sentence attention:** share=0.049, rank=8/8 -- how much attention mass landed on the sentence containing the model's own predicted answer, and its rank among all sentences (1 = most-attended sentence IS the answer sentence).

**Question 4:** What time period was the early Heian era?

**Gold answer:** 784–967

**Predicted answer:** 784–967 (f1=1.00, conf=0.985)

**Entropy:** sent_entropy_norm=0.922, tok_entropy_norm=0.625, combined=0.773 (sent_entropy=1.917, tok_entropy=3.480, num_sentences=8)

**Answer-sentence attention:** share=0.125, rank=6/8 -- how much attention mass landed on the sentence containing the model's own predicted answer, and its rank among all sentences (1 = most-attended sentence IS the answer sentence).

**Question 5:** Kanmu modeled his government after what Chinese capital?

**Gold answer:** Tang

**Predicted answer:** Tang (f1=1.00, conf=0.913)

**Entropy:** sent_entropy_norm=0.932, tok_entropy_norm=0.689, combined=0.810 (sent_entropy=1.937, tok_entropy=3.839, num_sentences=8)

**Answer-sentence attention:** share=0.191, rank=1/8 -- how much attention mass landed on the sentence containing the model's own predicted answer, and its rank among all sentences (1 = most-attended sentence IS the answer sentence).

**Probe question:** What is the main topic of the passage?

**Probe predicted answer:** improve his seat of government (conf=0.137)

**Probe entropy:** sent_entropy_norm=0.938, tok_entropy_norm=0.667, combined=0.803

### Passage 4

**Passage:**

> Switzerland has a dense network of cities, where large, medium and small cities are complementary. The plateau is very densely populated with about 450 people per km2 and the landscape continually shows signs of human presence. The weight of the largest metropolitan areas, which are Zürich, Geneva–Lausanne, Basel and Bern tend to increase. In international comparison the importance of these urban areas is stronger than their number of inhabitants suggests. In addition the two main centers of Zürich and Geneva are recognized for their particularly great quality of life.

(3/3 of this passage's questions were captured_correct -- only those are shown below)

**Question 1:** What is the population density of the plateau?

**Gold answer:** 450 people per km2

**Predicted answer:** 450 people per km2 (f1=1.00, conf=0.881)

**Entropy:** sent_entropy_norm=0.939, tok_entropy_norm=0.573, combined=0.756 (sent_entropy=1.511, tok_entropy=2.692, num_sentences=5)

**Answer-sentence attention:** share=0.223, rank=4/5 -- how much attention mass landed on the sentence containing the model's own predicted answer, and its rank among all sentences (1 = most-attended sentence IS the answer sentence).

**Question 2:** Which 2 centers are recognized for their particularly great quality of life?

**Gold answer:** Zürich and Geneva

**Predicted answer:** Zürich and Geneva (f1=1.00, conf=0.988)

**Entropy:** sent_entropy_norm=0.952, tok_entropy_norm=0.594, combined=0.773 (sent_entropy=1.532, tok_entropy=2.790, num_sentences=5)

**Answer-sentence attention:** share=0.370, rank=1/5 -- how much attention mass landed on the sentence containing the model's own predicted answer, and its rank among all sentences (1 = most-attended sentence IS the answer sentence).

**Question 3:** What does the weight of the largest metropolitan areas tend to do?

**Gold answer:** increase

**Predicted answer:** increase (f1=1.00, conf=0.933)

**Entropy:** sent_entropy_norm=0.995, tok_entropy_norm=0.564, combined=0.780 (sent_entropy=1.601, tok_entropy=2.653, num_sentences=5)

**Answer-sentence attention:** share=0.248, rank=1/5 -- how much attention mass landed on the sentence containing the model's own predicted answer, and its rank among all sentences (1 = most-attended sentence IS the answer sentence).

**Probe question:** What is the main topic of the passage?

**Probe predicted answer:** the landscape continually shows signs of human presence (conf=0.061)

**Probe entropy:** sent_entropy_norm=0.900, tok_entropy_norm=0.444, combined=0.672

### Passage 5

**Passage:**

> In 1937, IBM's tabulating equipment enabled organizations to process unprecedented amounts of data, its clients including the U.S. Government, during its first effort to maintain the employment records for 26 million people pursuant to the Social Security Act, and the Third Reich, largely through the German subsidiary Dehomag. During the Second World War the company produced small arms for the American war effort (M1 Carbine, and Browning Automatic Rifle). IBM provided translation services for the Nuremberg Trials. In 1947, IBM opened its first office in Bahrain, as well as an office in Saudi Arabia to service the needs of the Arabian-American Oil Company that would grow to become Saudi Business Machines (SBM).

(5/5 of this passage's questions were captured_correct -- only those are shown below)

**Question 1:** What what was the name of the subsidiary working in Germany during World War 2?

**Gold answer:** Dehomag

**Predicted answer:** Dehomag (f1=1.00, conf=1.000)

**Entropy:** sent_entropy_norm=0.770, tok_entropy_norm=0.611, combined=0.691 (sent_entropy=1.380, tok_entropy=3.023, num_sentences=6)

**Answer-sentence attention:** share=0.362, rank=1/6 -- how much attention mass landed on the sentence containing the model's own predicted answer, and its rank among all sentences (1 = most-attended sentence IS the answer sentence).

**Question 2:** Records for how many people were maintained by IBM in 1937?

**Gold answer:** 26 million

**Predicted answer:** 26 million (f1=1.00, conf=0.820)

**Entropy:** sent_entropy_norm=0.777, tok_entropy_norm=0.607, combined=0.692 (sent_entropy=1.392, tok_entropy=3.002, num_sentences=6)

**Answer-sentence attention:** share=0.365, rank=1/6 -- how much attention mass landed on the sentence containing the model's own predicted answer, and its rank among all sentences (1 = most-attended sentence IS the answer sentence).

**Question 3:** What service did IBM provide for the Nuremberg Trials?

**Gold answer:** translation services

**Predicted answer:** translation (f1=0.67, conf=0.505)

**Entropy:** sent_entropy_norm=0.641, tok_entropy_norm=0.521, combined=0.581 (sent_entropy=1.148, tok_entropy=2.578, num_sentences=6)

**Answer-sentence attention:** share=0.484, rank=1/6 -- how much attention mass landed on the sentence containing the model's own predicted answer, and its rank among all sentences (1 = most-attended sentence IS the answer sentence).

**Question 4:** What year did IBM open its first office in Bahrain?

**Gold answer:** 1947

**Predicted answer:** 1947 (f1=1.00, conf=1.000)

**Entropy:** sent_entropy_norm=0.734, tok_entropy_norm=0.567, combined=0.651 (sent_entropy=1.316, tok_entropy=2.806, num_sentences=6)

**Answer-sentence attention:** share=0.270, rank=3/6 -- how much attention mass landed on the sentence containing the model's own predicted answer, and its rank among all sentences (1 = most-attended sentence IS the answer sentence).

**Question 5:** What was the eventual name of the company that IBM operated in Saudi Arabia?

**Gold answer:** Saudi Business Machines

**Predicted answer:** Saudi Business Machines (f1=1.00, conf=0.631)

**Entropy:** sent_entropy_norm=0.754, tok_entropy_norm=0.556, combined=0.655 (sent_entropy=1.351, tok_entropy=2.754, num_sentences=6)

**Answer-sentence attention:** share=0.197, rank=3/6 -- how much attention mass landed on the sentence containing the model's own predicted answer, and its rank among all sentences (1 = most-attended sentence IS the answer sentence).

**Probe question:** What is the main topic of the passage?

**Probe predicted answer:** employment records for 26 million people pursuant to the Social Security Act (conf=0.107)

**Probe entropy:** sent_entropy_norm=0.629, tok_entropy_norm=0.412, combined=0.521

### Passage 6

**Passage:**

> The College Dropout was eventually issued by Roc-A-Fella in February 2004, shooting to number two on the Billboard 200 as his debut single, "Through the Wire" peaked at number fifteen on the Billboard Hot 100 chart for five weeks. "Slow Jamz", his second single featuring Twista and Jamie Foxx, became an even bigger success: it became the three musicians' first number one hit. The College Dropout received near-universal critical acclaim from contemporary music critics, was voted the top album of the year by two major music publications, and has consistently been ranked among the great hip-hop works and debut albums by artists. "Jesus Walks", the album's fourth single, perhaps exposed West to a wider audience; the song's subject matter concerns faith and Christianity. The song nevertheless reached the top 20 of the Billboard pop charts, despite industry executives' predictions that a song containing such blatant declarations of faith would never make it to radio. The College Dropout would eventually be certified triple platinum in the US, and garnered West 10 Grammy nominations, including Album of the Year, and Best Rap Album (which it received). During this period, West also founded GOOD Music, a record label and management company that would go on to house affiliate artists and producers, such as No I.D. and John Legend. At the time, the focal point of West's production style was the use of sped-up vocal samples from soul records. However, partly because of the acclaim of The College Dropout, such sampling had been much copied by others; with that overuse, and also because West felt he had become too dependent on the technique, he decided to find a new sound.

(3/5 of this passage's questions were captured_correct -- only those are shown below)

**Question 1:** What was the name of the single off the debut album that gave Kanye mainstream attention?

**Gold answer:** Jesus Walks

**Predicted answer:** Jesus Walks (f1=1.00, conf=0.594)

**Entropy:** sent_entropy_norm=0.869, tok_entropy_norm=0.718, combined=0.794 (sent_entropy=2.085, tok_entropy=4.186, num_sentences=11)

**Answer-sentence attention:** share=0.164, rank=2/11 -- how much attention mass landed on the sentence containing the model's own predicted answer, and its rank among all sentences (1 = most-attended sentence IS the answer sentence).

**Question 2:** What label did Kanye create following the success of his first album's release?

**Gold answer:** GOOD Music

**Predicted answer:** GOOD Music (f1=1.00, conf=0.997)

**Entropy:** sent_entropy_norm=0.893, tok_entropy_norm=0.732, combined=0.812 (sent_entropy=2.141, tok_entropy=4.269, num_sentences=11)

**Answer-sentence attention:** share=0.133, rank=3/11 -- how much attention mass landed on the sentence containing the model's own predicted answer, and its rank among all sentences (1 = most-attended sentence IS the answer sentence).

**Question 3:** When was The College Dropout finally released?

**Gold answer:** February 2004

**Predicted answer:** February 2004 (f1=1.00, conf=0.992)

**Entropy:** sent_entropy_norm=0.848, tok_entropy_norm=0.608, combined=0.728 (sent_entropy=2.034, tok_entropy=3.545, num_sentences=11)

**Answer-sentence attention:** share=0.098, rank=6/11 -- how much attention mass landed on the sentence containing the model's own predicted answer, and its rank among all sentences (1 = most-attended sentence IS the answer sentence).

**Probe question:** What is the main topic of the passage?

**Probe predicted answer:** faith and Christianity (conf=0.837)

**Probe entropy:** sent_entropy_norm=0.848, tok_entropy_norm=0.693, combined=0.770

Avg over 23 captured=Y question(s): sent_entropy_norm=0.884, tok_entropy_norm=0.621, combined=0.753 (probe combined avg=0.674); answer_sentence_attention_share=0.230, answer_sentence_rank=3.00 (n_located=23)

