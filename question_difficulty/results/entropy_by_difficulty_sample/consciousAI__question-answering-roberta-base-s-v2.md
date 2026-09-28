# Entropy and attention-shape metrics on captured-correct samples: `consciousAI/question-answering-roberta-base-s-v2`

layer=11, target 6 captured_correct passages per group (up to 400 candidates tried per group, up to 5 questions shown per passage). `captured_correct` = f1>=0.5 OR (recall_overlap>=0.8 AND pred<= 30% of passage). A passage is included if at least one of its real questions passes the gate -- these metrics are only a trustworthy difficulty signal on captured_correct=Y rows. `pr_norm` (participation ratio, normalized) is an alternative to entropy_norm that's robust to a noisy attention tail -- see the two summary tables below. `topK_mass` = cumulative attention mass on the K most-attended sentences/tokens. These are aggregate numbers only, meant to point you at which passages/questions to read manually below -- they don't replace manual review, since the EASY/MEDIUM/HARD label is passage-level (RACE, OneStopQA), not per-question.

## Summary (sentence-level)

| source | level | entropy_norm | pr_norm | top1_mass | top2_mass | top3_mass | n_questions | n_passages |
|---|---|---|---|---|---|---|---|---|
| RACE-middle | EASY | 0.816 | 0.492 | 0.181 | 0.347 | 0.506 | 8 | 6 |
| RACE-high | MEDIUM | 0.870 | 0.597 | 0.175 | 0.319 | 0.426 | 6 | 6 |
| RACE-C | HARD | 0.858 | 0.575 | 0.166 | 0.282 | 0.392 | 7 | 6 |
| OneStopQA | EASY | 0.884 | 0.693 | 0.286 | 0.498 | 0.685 | 6 | 6 |
| OneStopQA | MEDIUM | 0.830 | 0.627 | 0.351 | 0.623 | 0.861 | 6 | 6 |
| OneStopQA | HARD | 0.852 | 0.651 | 0.269 | 0.493 | 0.686 | 7 | 6 |
| SQuAD | N/A | 0.884 | 0.720 | 0.302 | 0.548 | 0.734 | 23 | 6 |

## Summary (token-level)

| source | level | entropy_norm | pr_norm | top5_mass | top10_mass | top15_mass | n_questions | n_passages |
|---|---|---|---|---|---|---|---|---|
| RACE-middle | EASY | 0.641 | 0.061 | 0.542 | 0.687 | 0.726 | 8 | 6 |
| RACE-high | MEDIUM | 0.661 | 0.055 | 0.399 | 0.643 | 0.685 | 6 | 6 |
| RACE-C | HARD | 0.657 | 0.054 | 0.422 | 0.634 | 0.701 | 7 | 6 |
| OneStopQA | EASY | 0.696 | 0.074 | 0.591 | 0.647 | 0.682 | 6 | 6 |
| OneStopQA | MEDIUM | 0.606 | 0.052 | 0.682 | 0.728 | 0.761 | 6 | 6 |
| OneStopQA | HARD | 0.656 | 0.068 | 0.585 | 0.679 | 0.718 | 7 | 6 |
| SQuAD | N/A | 0.621 | 0.048 | 0.641 | 0.700 | 0.735 | 23 | 6 |

## RACE-middle / EASY

6/6 captured (tried 7 candidates)

### Passage 1

**Passage:**

> When travelling.you are sure to try some exciting new kinds of food.The Wild Food Festival,in the town of Hokitika,the west of Coast of New Zealand,gives you the chance to try some strange food.It is a celebration of the areas special lifestyle and food.And it celebrates food that most people might not want to eat.It is held in March every year.
At the festival you will find huhu grubs and beetles on your plate.The festival also celebrates Maori food. the food of the traditional native people of New Nealand And visitors will eat the wild food with plenty of famous West Coast beer.What's more,there are three stages at the festival,where there is live music and entertainment an day long.
If you have the chance to travel to Hokitita during the Wild Food Festival,you should book a hotel before it begins.or you can choose to stay at local schools.A number of local schools become camping grounds over the weekend of the festival.You can also stay in Greymouth,because there are buses from Greymouth to the festival.

14 sentences, 225 tokens. 2/4 of this passage's questions were captured_correct -- only those are shown below, plus a fixed topic-probe question for comparison (no gold answer, so no f1).

| # | question | gold | pred | f1 | sent_entropy_norm | sent_pr_norm | tok_entropy_norm | tok_pr_norm |
|---|---|---|---|---|---|---|---|---|
| 1 | When is the Wild Food Festival held in the town of Hokitika every year? | In March. | March | 0.67 | 0.798 | 0.433 | 0.595 | 0.035 |
| 2 | What will you find on your plate at the festival? | Huhu grubs and beetles. | huhu grubs and beetles | 1.00 | 0.784 | 0.394 | 0.655 | 0.039 |
| probe | What is the main topic of the passage? | -- | food | -- | 0.779 | 0.414 | 0.588 | 0.034 |

### Passage 2

**Passage:**

> George Stephenson was born in 1781 in a poor family. He had to start work when he was only eight.When George was fourteen, he became his father's helper.He spent a lot of time learning about engines .And on holidays he often made one engine to pieces and studied each piece carefully.Soon he became a very good worker though he could not read or write.He began to learn English letters when he was seventeen years old.Every day after he did twelve hours of hard work, he walked a long way to have lessons from a young school teacher. On his eighteenth birthday,he wrote his own name for the first time in his life.George invented  many things in his life.The train was the greatest one among them.Today when we take trains from one place to another,we'll think of this great man---George Stephenson.

12 sentences, 172 tokens. 2/5 of this passage's questions were captured_correct -- only those are shown below, plus a fixed topic-probe question for comparison (no gold answer, so no f1).

| # | question | gold | pred | f1 | sent_entropy_norm | sent_pr_norm | tok_entropy_norm | tok_pr_norm |
|---|---|---|---|---|---|---|---|---|
| 1 | He spent a lot of time learning about engines and soon he became   _  . | a good worker | a very good worker | 0.86 | 0.787 | 0.446 | 0.574 | 0.041 |
| 2 | George Stephenson invented the   _  . | the train | The train | 1.00 | 0.664 | 0.289 | 0.498 | 0.028 |
| probe | What is the main topic of the passage? | -- | engines | -- | 0.720 | 0.350 | 0.577 | 0.034 |

### Passage 3

**Passage:**

> Some people think only school children do not agree with their parents, however, it is not true. Communication is a problem for parents and children of all ages. If it's hard for you to communicate with your parents, don't worry about it. Here are some advice for you to _ the generation gap  .
Don't argue with your parents. Don't get to your parents when you are angry. Your parents probably won't consider your ideas if you are shouting at them. And you can't express yourself well if you are angry. Go someplace to cool off. Make sure you understand why you are unhappy. Then think about what you want to say to your parents. If you don't think you can speak to them at the moment, try writing a letter.
Try to reach a compromise  . Perhaps you and your parents disagree on something. You can keep your disagreement and try your best to accept each other. Michael's mother didn't agree with him about buying a motorbike. They argued over it. But they finally came to a compromise. Michael bought the motorbike, but only drove it on certain days.
Of course, your parents might refuse to compromise on something. In these situations, it is especially important to show love and respect to them. Showing respect will keep your relationship  strong.
Talk about your values. The values of your parents are probably different from those of your own. Tell your parents what you care about, and why. Understanding your values might help them see your purposes in life.
A good relationship with your parents can make you a better and happier person. It is worth having a try!
,.

29 sentences, 335 tokens. 1/4 of this passage's questions were captured_correct -- only those are shown below, plus a fixed topic-probe question for comparison (no gold answer, so no f1).

| # | question | gold | pred | f1 | sent_entropy_norm | sent_pr_norm | tok_entropy_norm | tok_pr_norm |
|---|---|---|---|---|---|---|---|---|
| 1 | According to the passage who have a communication problem? | parents and children of all ages | parents and children of all ages | 1.00 | 0.873 | 0.516 | 0.757 | 0.099 |
| probe | What is the main topic of the passage? | -- | Communication | -- | 0.845 | 0.458 | 0.702 | 0.072 |

### Passage 4

**Passage:**

> Inventions named after people
Many new things are invented each year. Interestingly, some inventions are named after the people who invented them, making the inventions and inventors easier to be remembered.
The bowler hat is named after London hat-makers Thomas and William Bowler. The brothers received an order from Edward Coke, the younger brother of the 2ndEarl of Leicester. They were asked to design a hat for Coke's gamekeepers to protect _ heads from branches while on horseback. Later, the stylish hat became popular in Europe and the United States.
Another invention that is named after its inventor is Braille, a writing system used by blind people. French educator Louis Braille developed a new system of reading and writing after learning the cryptography of French Captain Charles Barbier during the war. The captain had come up with a code of dots on paper that allowed soldiers to communicate in the dark.
The diesel engine is also named after its inventor----German engineer Rudolf Diesel. After a few dangerous tests, he invented a new and more efficient engine in 1892 and the engine was later called the diesel engine. The engines were widely used in buses, trucks, trains and ships, and Rudolf Diesel became a millionaire.

12 sentences, 249 tokens. 1/4 of this passage's questions were captured_correct -- only those are shown below, plus a fixed topic-probe question for comparison (no gold answer, so no f1).

| # | question | gold | pred | f1 | sent_entropy_norm | sent_pr_norm | tok_entropy_norm | tok_pr_norm |
|---|---|---|---|---|---|---|---|---|
| 1 | Who invented the bowler hat? | Thomas and William Bowler. | Thomas and William Bowler | 1.00 | 0.981 | 0.919 | 0.756 | 0.105 |
| probe | What is the main topic of the passage? | -- | making the inventions and inventors easier to be remembered | -- | 0.936 | 0.785 | 0.663 | 0.064 |

### Passage 5

**Passage:**

> Whenever there's a terrible storm, there are plenty of sad stories. Many people lost their houses, their cars and their pets. But sometimes these stories have happy endings too.
A family in New Jersey, US, had a cat named Vivien. She is very smart. She could even draw with her wet paws on the floor. They all love her very much. When hurricane   Sandy hit America in October, they moved to a safer place-13km away from home. Sadly Vivien went missing. The whole family were worried about her. They put up posters on the Internet to look for her. All the nine family members searched wherever they thought she could stay, but they didn't find it. The family thought Vivien was gone forever.
But six months later, Vivien showed up at their house, according to Yahoo News. They considered her return as a wonder. No one can be sure where Vivien was for all that time.
Many animals are good at finding their way home. People usually say that dogs and cats find their way home through using their sense of smell.
But that doesn't explain how Vivien found her way back. Hurricane Sandy blew away the normal smells of home.
"I wish she could talk," said her owner.

21 sentences, 259 tokens. 1/3 of this passage's questions were captured_correct -- only those are shown below, plus a fixed topic-probe question for comparison (no gold answer, so no f1).

| # | question | gold | pred | f1 | sent_entropy_norm | sent_pr_norm | tok_entropy_norm | tok_pr_norm |
|---|---|---|---|---|---|---|---|---|
| 1 | From the passage we know Sandy is a | hurricane | hurricane | 1.00 | 0.833 | 0.500 | 0.622 | 0.059 |
| probe | What is the main topic of the passage? | -- | sad stories | -- | 0.878 | 0.592 | 0.702 | 0.082 |

### Passage 6

**Passage:**

> Dear Jenny,
I'm sorry you're sick and can't come to school this week. Let me tell you what you have on Monday next week.
We have math at 8:00. How boring! Then we have English. That's interesting. I know you love English. Next is P.E.. The fourth lesson is art. That's my favorite subject! In the afternoon, we have history with Mr. Chen. He is fun but history isn't.  Then at 2:00 we have science with Miss Jones. You know how strict she is. I don't like her. Our last class of the day is math.
Yours,
Ben

19 sentences, 137 tokens. 1/5 of this passage's questions were captured_correct -- only those are shown below, plus a fixed topic-probe question for comparison (no gold answer, so no f1).

| # | question | gold | pred | f1 | sent_entropy_norm | sent_pr_norm | tok_entropy_norm | tok_pr_norm |
|---|---|---|---|---|---|---|---|---|
| 1 | Their first lesson is   _  . | math | math | 1.00 | 0.806 | 0.435 | 0.668 | 0.084 |
| probe | What is the main topic of the passage? | -- | math | -- | 0.788 | 0.420 | 0.683 | 0.093 |

Avg over 8 captured=Y question(s) -- sentence: entropy_norm=0.816, pr_norm=0.492, top1=0.181, top2=0.347, top3=0.506; token: entropy_norm=0.641, pr_norm=0.061, top5=0.542, top10=0.687, top15=0.726

## RACE-high / MEDIUM

6/6 captured (tried 24 candidates)

### Passage 1

**Passage:**

> The Sapporo Snow Festival (Sapporo Yuki-matsuri) is a festival being held every year in Sapporo, Japan for over seven days in February. Presently, Odori Park, Susukino, and Tsudome are the main places of the festival. The 2013 Yuki-matsuri dates are February 5 to 11. 
    The festival is one of Japan's largest and most typical winter events. In 2007 (58th Festival), about two million people visited Sapporo to see the hundreds of floating statues and ice sculptures at the Odori Park and Susukino sites, in central Sapporo, and at the Satoland site. The festival is thought to be a chance for promoting international relations. International Snow Sculpture Contest has been held at the Odori Park site since 1974, and 14 teams from various areas of the world participated in 2008. 
   The subject of the statues differs and often shows an event, famous building or person from the previous year. For example, in 2004, there were statues of Hideki Matsui, the famous baseball player who at that time played for the New York Yankees. A number of stages made out of snow are also constructed and some events including musical performances are held. At the Satoland site, visitors can enjoy long snow and ice slides as well as a huge maze  made of snow. Visitors can also enjoy a variety of local foods from all over Hokkaido at the Odori Park and Satoland sites, such as fresh seafood, potatoes and corn, and fresh dairy products. 
    Every year the number of Statues displayed is around 400 in total. In 2007, ther were 307 statues created in the Odori Park site, 32 in the Satoland site and 100 in the Susukino site. The best place to view the creations is from the TV Tower at the Odori Park site. Most of the statues are lighted in the evening. The Sapporo Snow Festival Museum is placed in the Hitsujigaoka observation hill  in Toyohira-ku, and displays historical materials and media of the festival.

17 sentences, 436 tokens. 1/3 of this passage's questions were captured_correct -- only those are shown below, plus a fixed topic-probe question for comparison (no gold answer, so no f1).

| # | question | gold | pred | f1 | sent_entropy_norm | sent_pr_norm | tok_entropy_norm | tok_pr_norm |
|---|---|---|---|---|---|---|---|---|
| 1 | The Sapporo Snow Festival 2013 will start from   _  . | February 5 | February | 0.67 | 0.898 | 0.686 | 0.556 | 0.034 |
| probe | What is the main topic of the passage? | -- | promoting international relations | -- | 0.902 | 0.648 | 0.693 | 0.051 |

### Passage 2

**Passage:**

> SHANGHAI, June 7(AP)--A 16-year-old girl's suicide after she was barred from a key exam draw attention to increasing worries over academic pressures, as millions of Chinese students began annual college entrance tests on Wednesday. 
The three-day exam, viewed as important to future career and financial success, has a record 9.5 million high school students across prefix = st1 /Chinacompeting for just 2.6 million university places. For kids and parents alike, it's a difficulty that experts say causes extreme emotional distress. "Pressure from study and exams is a top reason for psychological problems among Chinese youth," said Jin Wuguan, director of the Youth Psychological Counseling Center at Shanghai'sRuijinHospital. 
In China's increasingly success oriented, pressure-cooker cities, academic stress is seen as a rising cause of youth suicides and even murders of parents by children who are driven crazy by intolerable pressure to perform. 
According to her family and newspaper accounts, 16-year-old Wu Wenwen drowned herself after she was stopped at the exam room door because her hair wasn't tied back as her school required. Returning in tied hair, she was then told the end-of-term exam had already started and she was too late to take it. In tears, Wu called her mother, and then disappeared. Her body was found the same night in a nearby lake. 
China doesn't keep comprehensive statistics on student suicides, but Jin said health care professionals see the problem worsening, even among elementary students. Most Chinese schools still lack advisers and teachers receive little training in spotting symptoms of emotional distress, Jin said. Parents are little help, often piling on pressure while ignoring their children's emotional development, he said. "It's a basic unwillingness or inability to recognize and deal with with emotional problems," Jin said. 
Wang Yufeng, of Peking University's Institute of Mental, estimates the rate of emotional disorders such as depression among Chinese students under age 17 at up to 32 percent , a total of 30 million students. 
Others say that figure may be as high as 50 percent. A survey last year by the government's China Youth and ChildrenResearchCentershowed 57.6 percent of students felt highly distressed by academic pressures.

19 sentences, 470 tokens. 1/3 of this passage's questions were captured_correct -- only those are shown below, plus a fixed topic-probe question for comparison (no gold answer, so no f1).

| # | question | gold | pred | f1 | sent_entropy_norm | sent_pr_norm | tok_entropy_norm | tok_pr_norm |
|---|---|---|---|---|---|---|---|---|
| 1 | Where will we most probably find the article? | In a newspaper. | newspaper | 0.50 | 0.901 | 0.644 | 0.720 | 0.064 |
| probe | What is the main topic of the passage? | -- | academic pressures | -- | 0.872 | 0.556 | 0.689 | 0.060 |

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

14 sentences, 303 tokens. 1/3 of this passage's questions were captured_correct -- only those are shown below, plus a fixed topic-probe question for comparison (no gold answer, so no f1).

| # | question | gold | pred | f1 | sent_entropy_norm | sent_pr_norm | tok_entropy_norm | tok_pr_norm |
|---|---|---|---|---|---|---|---|---|
| 1 | Tom is a   _  , crazy about selfies. | fanboy | Fangirl" or "fanboy | 0.50 | 0.873 | 0.586 | 0.667 | 0.054 |
| probe | What is the main topic of the passage? | -- | M | -- | 0.876 | 0.613 | 0.592 | 0.041 |

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

7 sentences, 235 tokens. 1/2 of this passage's questions were captured_correct -- only those are shown below, plus a fixed topic-probe question for comparison (no gold answer, so no f1).

| # | question | gold | pred | f1 | sent_entropy_norm | sent_pr_norm | tok_entropy_norm | tok_pr_norm |
|---|---|---|---|---|---|---|---|---|
| 1 | Which is most probably the News Channel? | Channel9. | Channel 9 | 0.00 | 0.751 | 0.428 | 0.547 | 0.021 |
| probe | What is the main topic of the passage? | -- | Pollution in the Mississippi | -- | 0.792 | 0.480 | 0.584 | 0.026 |

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

20 sentences, 448 tokens. 1/4 of this passage's questions were captured_correct -- only those are shown below, plus a fixed topic-probe question for comparison (no gold answer, so no f1).

| # | question | gold | pred | f1 | sent_entropy_norm | sent_pr_norm | tok_entropy_norm | tok_pr_norm |
|---|---|---|---|---|---|---|---|---|
| 1 | We can infer that the Independent's "i" is aimed at   _  . | young readers | young people | 0.50 | 0.865 | 0.528 | 0.658 | 0.045 |
| probe | What is the main topic of the passage? | -- | Most "i" products are targeted at young people and considering the major readers of Independent's "i", it's no surprise that they've selected this fashionable name. 
But it's hard to see what's so special about the letter "i". Why not use "a", "b", or "c" instead? According to Tony Thorne, head of the Language Center at King's College, London, "i" works because its meaning has become ambiguous. When Apple uses "i", no one knows whether it means Internet, information, individual or interactive, Thorne told BBC Magazines. "Even when Apple created the iPod, it seems it didn't have one clear definition," he says. 
"However, thanks to Apple, the term is now associated with portability." adds Thorne.
Clearly the letter "i" also agrees with the idea that the Western World is centered on the individual. Each person believes they have their own needs | -- | 0.877 | 0.590 | 0.623 | 0.040 |

### Passage 6

**Passage:**

> Our flat was on the fifth floor but you could still hear the roar of the ocean and see the stars at night. I used to take long walks along the water. The food in town was delicious and the people were very friendly. The area was very quiet and peaceful, and fairly deserted. 
The last evening of our vacation, however, we all heard strange footsteps following closely behind us as we were walking up to our flat in the holiday centre. We turned around and noticed a fairly young man moving very rapidly across the beach and getting closer to us. He was tall and wore a baseball cap. We couldn't see his face and he was approaching us very rapidly. The man's actions made my dad very nervous. Dad warned us that we'd better try to make it to our flat as quickly as possible. I didn't like my dad's voice; I could hear fear in it. It was late and we were all alone. We didn't have any cell phones on us. I never saw Dad as worried as he was then and I knew that something was terribly wrong. The sense of fear started to overwhelm Mom and me. We had had such a good time in town. Now, the night was rapidly turning into a dangerous situation. 
We could hear the man's footsteps getting closer. Dad's face was almost pale. The so-called intruder   had moved nearer and nearer when all of a sudden, the nearby vending  machine started going crazy and spitting out cans of soda! The noise actually scared the intruder and he ran out of sight. My parents were shaking, but we all turned around to see who had put money into the vending machine downstairs, and actually saved us, but no one was around at all. Not a soul. 
It's one vacation I will never forget.

24 sentences, 367 tokens. 1/2 of this passage's questions were captured_correct -- only those are shown below, plus a fixed topic-probe question for comparison (no gold answer, so no f1).

| # | question | gold | pred | f1 | sent_entropy_norm | sent_pr_norm | tok_entropy_norm | tok_pr_norm |
|---|---|---|---|---|---|---|---|---|
| 1 | What helped them get rid of the trouble? | The noise from the vending machine. | soda! The noise actually scared the intruder and he ran out of sight. My parents were shaking, but we all turned around to see who had put money into the vending machine | 0.21 | 0.935 | 0.712 | 0.819 | 0.111 |
| probe | What is the main topic of the passage? | -- | water | -- | 0.888 | 0.582 | 0.723 | 0.065 |

Avg over 6 captured=Y question(s) -- sentence: entropy_norm=0.870, pr_norm=0.597, top1=0.175, top2=0.319, top3=0.426; token: entropy_norm=0.661, pr_norm=0.055, top5=0.399, top10=0.643, top15=0.685

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

43 sentences, 458 tokens. 1/4 of this passage's questions were captured_correct -- only those are shown below, plus a fixed topic-probe question for comparison (no gold answer, so no f1).

| # | question | gold | pred | f1 | sent_entropy_norm | sent_pr_norm | tok_entropy_norm | tok_pr_norm |
|---|---|---|---|---|---|---|---|---|
| 1 | Lack of Vitamin A will lead to _ . | night blindness | blindness | 0.67 | 0.700 | 0.226 | 0.530 | 0.024 |
| probe | What is the main topic of the passage? | -- | Vitamins are important to our health | -- | 0.730 | 0.228 | 0.584 | 0.025 |

### Passage 2

**Passage:**

> Anne Whitney, a sophomore at Colorado State University, first had a problem taking tests when she began college. "I was always well prepared for my tests. Sometimes I studied for weeks before a test. Yet I would go in to take the test, only to find I could not answer the questions correctly. I would blank out because of nervousness and fear. I couldn't think of the answer. My low grades on the tests did not show what I knew to the teacher." Another student in microbiology and similar experiences. He said, "My first chemistry test was very difficult. Then, on the second test, sat down to take it, and I was so nervous that I was shaking. My hands were moving up and down so quickly that it was hard to hold my pencil. I knew the materical and I knew the answers. Yet I couldn't even writen them down!"
These two young students were experiencing something called test anxiety. Because a student worries and it uneasy about a test, his or her mind does not work as well as it usually does. The student can't write or think clearly because of the extreme tension and nervousness. Although poor grades are often a result of poor study habits, sometimes test anxiety causes the low grades. Recently, test anxiety had been recognized as a real problem, not just an excuse or a false explanation of lazy students.
Special university counseling courses try to help students. In these courses, counselors try to help students by teaching them how to manage test anxiety. At some universities, students take tests to measure their anxiety. If the tests show their anxiety is high, the students can take short courses to help them deal with their tension. These courses teach students how to relax their bodies. Students are trained to become calm in bery tense situations. By controlling their nervousness, they can let their minds work at ease. Learned information then comes out without difficulty on a test.
An expert at the University of California explains. "With almost all students, relaxation and less stress are felt after taking out program. Most of then experience better control during their tests. Almost all have some improvement. With some, the improvement is very great."

32 sentences, 440 tokens. 1/5 of this passage's questions were captured_correct -- only those are shown below, plus a fixed topic-probe question for comparison (no gold answer, so no f1).

| # | question | gold | pred | f1 | sent_entropy_norm | sent_pr_norm | tok_entropy_norm | tok_pr_norm |
|---|---|---|---|---|---|---|---|---|
| 1 | Test anxiety has been recognized as _ . | a real problem | a real problem | 1.00 | 0.757 | 0.306 | 0.603 | 0.049 |
| probe | What is the main topic of the passage? | -- | chemistry | -- | 0.805 | 0.409 | 0.611 | 0.042 |

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

25 sentences, 499 tokens. 1/3 of this passage's questions were captured_correct -- only those are shown below, plus a fixed topic-probe question for comparison (no gold answer, so no f1).

| # | question | gold | pred | f1 | sent_entropy_norm | sent_pr_norm | tok_entropy_norm | tok_pr_norm |
|---|---|---|---|---|---|---|---|---|
| 1 | What did the Health Minister propose? | to allocate the patients to doctors on a list basis | that patients be allocated to doctors on a list basis | 0.70 | 0.812 | 0.449 | 0.715 | 0.060 |
| probe | What is the main topic of the passage? | -- | Health Minister Nicola Roxon's latest proposal that patients be allocated to doctors on a list basis is straight out of the playbook of Britain's National Health Service.
Let's think about this from the patient's point of view | -- | 0.794 | 0.450 | 0.609 | 0.041 |

### Passage 4

**Passage:**

> The English policeman has several nicknames but the most frequently used are "copper" and "bobby". The first name comes from the verb "to cop " (which is also slang ) , meaning " to take " or " to capture ", and the second comes from the first name of Sir Robert Peel, the nineteenth-century politician, who was the founder of the police force as we know it today. An early nickname for the policeman was "peeler", but this one has died out.
Whatever we may call them, the general opinion of the police seems to be a favorable one; except, of course, among the criminal part of the community where the police are given more derogatory nicknames which originated in America, such as "fuzz" or "pig". Visitors to England seem nearly always to be very impressed by the English police. It has, in fact, become a standing joke that the visitor to Britain, when asked for his views of the country, will always say, at some point or other, "I think your policemen are wonderful. "
Well, the British bobby may not always be wonderful but he is usually a very friendly and helpful sort of character. A music-hall song of some years ago was called "If You Want To Know The Time, Ask A Policeman". Nowadays, most people own watches but they still seem to find plenty of other questions to ask the policeman. In London, the policemen spend so much of their time directing visitors about the city that one wonders how they ever find time to do anything else!
Two things are immediately noticeable to the stranger when he sees an English policeman for the first time. The first is that he does not carry a pistol and the second is that he wears a very distinctive type of headgear, the policeman's helmet. His helmet, together with his height, enable an English policeman to be seen from a considerable distance, a fact that is not without its usefulness. From time to time it is suggested that the policeman should be given a pistol and that his helmet should be taken from him, but both these suggestions are resisted by the majority of the public and the police themselves.

14 sentences, 440 tokens. 1/4 of this passage's questions were captured_correct -- only those are shown below, plus a fixed topic-probe question for comparison (no gold answer, so no f1).

| # | question | gold | pred | f1 | sent_entropy_norm | sent_pr_norm | tok_entropy_norm | tok_pr_norm |
|---|---|---|---|---|---|---|---|---|
| 1 | Nowadays British people call the policeman _ . | bobby | bobby | 1.00 | 0.924 | 0.714 | 0.668 | 0.037 |
| probe | What is the main topic of the passage? | -- | criminal part of the community | -- | 0.901 | 0.664 | 0.635 | 0.032 |

### Passage 5

**Passage:**

> As supplier of most of the food we eat and of raw materials for many industrial processes, agriculture is clearly an important area of the economy. But the industrial performance of agriculture is even more important than this. For in nations where the productivity of farmers is low, most of the working population is needed to raise food and few people are available for production of investment goods or for other activities required for economic growth. Indeed, one of the factors related most closely to the per capital income  of a nation is the fraction of its population engaged in farming. In the poorest nations of the world more than half of the population lives on farms. This compares sharply with less than 10 per cent in Western Europe and less than 4 per cent in the United States.
In short, the course of economic development in general depends in a fundamental way on the performance of farmers. This performance in turn, depends on how agriculture is organized and on the economic environment, or market structure, within which it function. In the following pages the performance of American agriculture is examined. It is appropriate to begin with a conversation of its market structure.

10 sentences, 222 tokens. 2/5 of this passage's questions were captured_correct -- only those are shown below, plus a fixed topic-probe question for comparison (no gold answer, so no f1).

| # | question | gold | pred | f1 | sent_entropy_norm | sent_pr_norm | tok_entropy_norm | tok_pr_norm |
|---|---|---|---|---|---|---|---|---|
| 1 | What is most important to agriculture is ________. | its industrial performance | industrial performance | 0.80 | 0.955 | 0.826 | 0.701 | 0.073 |
| 2 | The performance of farmers essentially determines ________. | the general development of economy | the course of economic development | 0.60 | 0.924 | 0.731 | 0.668 | 0.063 |
| probe | What is the main topic of the passage? | -- | its market structure | -- | 0.972 | 0.888 | 0.674 | 0.072 |

### Passage 6

**Passage:**

> The biggest safety threat facing airlines today may not be a terrorist with a gun, but the man with the portable computer in business class. In the last 15 years, pilots have reported well over 100 incidents that could have been caused by electromagnetic interference. The source of this interference remains unconfirmed, but increasingly, experts are pointing the blame at portable electronic device such as portable computers, radio and cassette players and mobile telephones.
RTCA, an organization which advises the aviation  industry, has recommended that all airlines ban  such devices from being used during "critical" stages of flight, particularly take-off and landing. Some experts have gone further, calling for a total ban during all flights. Currently, rules on using these devices are left up to individual airlines. And although some airlines prohibit passengers from using such equipment during take-off and landing, most are reluctant to enforce a total ban, given that many passengers want to work during flights.
The difficulty is predicting how electromagnetic fields might affect an aircraft's computers. Experts know that portable device emit radiation which affects those wavelengths which aircraft use for navigation and communication. But, because they have not been able to reproduce these effects in a laboratory, they have no way of knowing whether the interference might be dangerous or not.
The fact that aircraft may be vulnerable  to interference raises the risk that terrorists may use radio systems in order to damage navigation equipment. As worrying, though, is the passenger who can't hear the instructions to turn off his radio because the music's too loud.

12 sentences, 308 tokens. 1/5 of this passage's questions were captured_correct -- only those are shown below, plus a fixed topic-probe question for comparison (no gold answer, so no f1).

| # | question | gold | pred | f1 | sent_entropy_norm | sent_pr_norm | tok_entropy_norm | tok_pr_norm |
|---|---|---|---|---|---|---|---|---|
| 1 | Why is it difficult to predict the possible effects of electromagnetic fields on an airplane's computers? | Because research scientists have not been able to produce the same effects in labs. | because they have not been able to reproduce these effects in a laboratory | 0.59 | 0.932 | 0.776 | 0.714 | 0.075 |
| probe | What is the main topic of the passage? | -- | the man with the portable computer in business class | -- | 0.904 | 0.668 | 0.690 | 0.069 |

Avg over 7 captured=Y question(s) -- sentence: entropy_norm=0.858, pr_norm=0.575, top1=0.166, top2=0.282, top3=0.392; token: entropy_norm=0.657, pr_norm=0.054, top5=0.422, top10=0.634, top15=0.701

## OneStopQA / EASY

6/6 captured (tried 11 candidates)

### Passage 1

**Passage:**

> After two years of successful ads with cute animals – a bear and hare, then a penguin – this time, the story is about a young girl, Lily, who sees an old man living in a small wooden house on the moon through her telescope. The girl first tries to send him a letter and a note via bow and arrow. Then, she floats him a present of a telescope tied to balloons. This finally allows them to make contact. The ad’s message is: “Show someone they’re loved this Christmas.” This is similar to Age UK’s campaign: “No one should have no one at Christmas.” Profits from three products – a mug, gift tag and card – will go to the charity. Rachel Swift, head of marketing at John Lewis, said that people talk about charities at Christmas and the ad makes you think about someone who lives on your street that might not see anybody.

8 sentences, 194 tokens. 1/3 of this passage's questions were captured_correct -- only those are shown below, plus a fixed topic-probe question for comparison (no gold answer, so no f1).

| # | question | gold | pred | f1 | sent_entropy_norm | sent_pr_norm | tok_entropy_norm | tok_pr_norm |
|---|---|---|---|---|---|---|---|---|
| 1 | Who does Rachel Swift work for? | John Lewis | John Lewis | 1.00 | 0.859 | 0.619 | 0.701 | 0.073 |
| probe | What is the main topic of the passage? | -- | a young girl, Lily, who sees an old man living in a small wooden house on the moon through her telescope | -- | 0.757 | 0.469 | 0.589 | 0.042 |

### Passage 2

**Passage:**

> Benjamin Carle is 96.9% made in France, even his underpants and socks. Six Ikea forks, a Chinese guitar and some wall paint stopped him being called 100% French, but nobody is perfect. Carle, 26, decided, in 2013, to see if it was possible to live using only French-made products for ten months as part of a television documentary. He got the idea after the Minister for Economic Renewal, Arnaud Montebourg, asked the French people to buy French products. For the experiment, Carle had to give up his smartphone, television, refrigerator (all made in China); his glasses (Italian); his morning coffee (Guatemalan) and his favourite David Bowie music (British). It is lucky that his girlfriend, Anaïs, and cat, Loon, are both French, so he didn’t have to give them up.

7 sentences, 183 tokens. 1/3 of this passage's questions were captured_correct -- only those are shown below, plus a fixed topic-probe question for comparison (no gold answer, so no f1).

| # | question | gold | pred | f1 | sent_entropy_norm | sent_pr_norm | tok_entropy_norm | tok_pr_norm |
|---|---|---|---|---|---|---|---|---|
| 1 | What did Arnaud Montebourgh do? | He asked people to buy products made in France | asked the French people to buy French products | 0.59 | 0.870 | 0.689 | 0.722 | 0.073 |
| probe | What is the main topic of the passage? | -- | nobody is perfect. Carle, 26, decided, in 2013, to see if it was possible to live using only French-made products for ten months as part of a television documentary | -- | 0.840 | 0.664 | 0.600 | 0.044 |

### Passage 3

**Passage:**

> The department store John Lewis has a 2015 Christmas advertisement. The ad shows a lonely old man who lives on the moon. The ad, which for many people shows that the Christmas shopping season has begun, aims to raise hundreds of thousands of pounds for the charity Age UK. John Lewis will also encourage staff and customers to care for elderly people who might be alone over the holiday. The department store has spent £7 million on a campaign that includes the TV ad, a smartphone game and merchandise, including glow-in-the-dark pyjamas. It will also build areas that look like the surface of the moon in 11 of its stores.

6 sentences, 129 tokens. 1/3 of this passage's questions were captured_correct -- only those are shown below, plus a fixed topic-probe question for comparison (no gold answer, so no f1).

| # | question | gold | pred | f1 | sent_entropy_norm | sent_pr_norm | tok_entropy_norm | tok_pr_norm |
|---|---|---|---|---|---|---|---|---|
| 1 | Where does the lonely old man appear in John Lewis’s advertisement? | On the moon | on the moon | 1.00 | 0.883 | 0.689 | 0.704 | 0.086 |
| probe | What is the main topic of the passage? | -- | Christmas advertisement. The ad shows a lonely old man who lives on the moon. The ad, which for many people shows that the Christmas shopping season has begun, aims to raise hundreds of thousands of pounds for the charity Age UK | -- | 0.847 | 0.657 | 0.520 | 0.042 |

### Passage 4

**Passage:**

> Autism is a disorder that one in 100 people have. It affects people in different ways, but causes difficulties in social interaction and communication. So far, there is no effective treatment for the social problems that autism causes. Researchers at Yale have studied the brain chemical oxytocin. They say it is a possible treatment for the social problems caused by autism because it plays an important role in bonding and trust. But not all results are positive: one recent study found no significant benefit for young people who took the chemical for several days. But Pelphrey said oxytocin might help the brain learn from social interactions; it would work best when used together with therapies that encourage people with autism to interact more socially, he said.

7 sentences, 145 tokens. 1/3 of this passage's questions were captured_correct -- only those are shown below, plus a fixed topic-probe question for comparison (no gold answer, so no f1).

| # | question | gold | pred | f1 | sent_entropy_norm | sent_pr_norm | tok_entropy_norm | tok_pr_norm |
|---|---|---|---|---|---|---|---|---|
| 1 | How were the social aspects of autism effectively treated before the time this article was written? | There was no effective treatment available | there is no effective treatment | 0.73 | 0.893 | 0.714 | 0.683 | 0.076 |
| probe | What is the main topic of the passage? | -- | Autism | -- | 0.828 | 0.612 | 0.541 | 0.049 |

### Passage 5

**Passage:**

> Scientists must find new names for the elements but, also, they must suggest two-letter symbols for the elements. When IUPAC has received the researchers’ suggestions, they will tell the public so that people can comment on the names. That allows scientists and others to find any problems with the names. In 1996, someone suggested the symbol Cp for copernicium, or element 112, but it was changed to Cn, when scientists complained that Cp was already the symbol for another substance.

4 sentences, 103 tokens. 1/3 of this passage's questions were captured_correct -- only those are shown below, plus a fixed topic-probe question for comparison (no gold answer, so no f1).

| # | question | gold | pred | f1 | sent_entropy_norm | sent_pr_norm | tok_entropy_norm | tok_pr_norm |
|---|---|---|---|---|---|---|---|---|
| 1 | Why was the suggestion to use symbol Cp for copernicium not accepted? | The symbol was already being used for a different substance | Cp was already the symbol for another substance | 0.67 | 0.883 | 0.702 | 0.670 | 0.072 |
| probe | What is the main topic of the passage? | -- | Scientists must find new names for the elements but, also, they must suggest two-letter symbols for the elements. When IUPAC has received the researchers’ suggestions, they will tell the public so that people can comment on the names. That allows scientists and others to find any problems with the names | -- | 0.851 | 0.689 | 0.481 | 0.038 |

### Passage 6

**Passage:**

> Most of our customers are “baby boomers who want to have the cycling experience they had as a kid,” says Pedego’s Don DiCostanza. “The main reason they stopped riding bikes was because of hills.” Pedego has opened nearly 60 stores in the US. ElectroBike has 30 stores in Mexico. It opened its first American store in Venice Beach, California in 2014 and hopes to have 25 US stores in a year. CEO Craig Anderson says: “We want to help reduce traffic, help reduce our carbon footprint and encourage a healthy lifestyle.” He tells customers: “Ride this bike once and try not to smile.” Startups like Pedego and ElectroBike will have to compete with big companies like Trek, Currie, and Accell - the market leader in e-bikes in Europe. Accell owns the Raleigh brand, as well as Haibike, an award-winning German electric bike.

9 sentences, 204 tokens. 1/3 of this passage's questions were captured_correct -- only those are shown below, plus a fixed topic-probe question for comparison (no gold answer, so no f1).

| # | question | gold | pred | f1 | sent_entropy_norm | sent_pr_norm | tok_entropy_norm | tok_pr_norm |
|---|---|---|---|---|---|---|---|---|
| 1 | What does ElectroBike hope to accomplish within a year of opening its store in Venice Beach? | Have 25 stores in the US | hopes to have 25 US stores in a year | 0.67 | 0.919 | 0.742 | 0.698 | 0.065 |
| probe | What is the main topic of the passage? | -- | reduce traffic | -- | 0.926 | 0.782 | 0.624 | 0.054 |

Avg over 6 captured=Y question(s) -- sentence: entropy_norm=0.884, pr_norm=0.693, top1=0.286, top2=0.498, top3=0.685; token: entropy_norm=0.696, pr_norm=0.074, top5=0.591, top10=0.647, top15=0.682

## OneStopQA / MEDIUM

6/6 captured (tried 9 candidates)

### Passage 1

**Passage:**

> The potential for solar power from the desert has been known for decades. In the days after the Chernobyl nuclear accident in 1986, the German particle physicist Gerhard Knies calculated that the world’s deserts receive enough energy in a few hours to provide power for all the people in the world for a whole year. But the challenge is to capture that energy and take it to where it is needed. Experts say that solar energy will make up a third of Morocco’s renewable energy supply by 2020. Wind and hydro will make up the other two-thirds. “We are very proud of this project,” Morocco’s environment minister, Hakima el-Haite said. “I think it is the most important solar plant in the world.”

8 sentences, 158 tokens. 1/3 of this passage's questions were captured_correct -- only those are shown below, plus a fixed topic-probe question for comparison (no gold answer, so no f1).

| # | question | gold | pred | f1 | sent_entropy_norm | sent_pr_norm | tok_entropy_norm | tok_pr_norm |
|---|---|---|---|---|---|---|---|---|
| 1 | By 2020, two-thirds of Morocco’s renewable energy will be ... | Hydro and wind energy | Wind and hydro | 0.86 | 0.792 | 0.550 | 0.536 | 0.039 |
| probe | What is the main topic of the passage? | -- | the challenge is to capture that energy and take it to where it is needed | -- | 0.793 | 0.539 | 0.570 | 0.045 |

### Passage 2

**Passage:**

> Vienna is the world’s best city to live in, Baghdad is the worst and London, Paris and New York do not even enter the top 35, according to international research into quality of life. German-speaking cities dominate the rankings in the 18th Mercer Quality of Life study, with Vienna joined by Zurich, Munich, Dusseldorf and Frankfurt in the top seven. Paris has dropped down the table – it has fallen ten places to 37th, just ahead of London at 39th, mostly because of the terrorist attacks on the city. The study examined social and economic conditions, health, education, housing and the environment. It is used by big companies to decide where they should open offices and factories and how much they should pay staff.

5 sentences, 153 tokens. 1/3 of this passage's questions were captured_correct -- only those are shown below, plus a fixed topic-probe question for comparison (no gold answer, so no f1).

| # | question | gold | pred | f1 | sent_entropy_norm | sent_pr_norm | tok_entropy_norm | tok_pr_norm |
|---|---|---|---|---|---|---|---|---|
| 1 | What is the ranking of Paris in the 18th Mercer Quality of Life study? | 37th | 37th | 1.00 | 0.858 | 0.650 | 0.574 | 0.038 |
| probe | What is the main topic of the passage? | -- | The study examined social and economic conditions, health, education, housing and the environment | -- | 0.810 | 0.569 | 0.558 | 0.040 |

### Passage 3

**Passage:**

> According to American researchers, a nasal spray containing the ‘Love hormone’ oxytocin could help children with autism behave more normally in social situations. Scans of autistic children showed that a single dose of the chemical improved brain responses to facial expressions. This is something that could make social interactions feel more natural and rewarding for them. The researchers said oxytocin might increase the success of behavioral therapies that are already used to help people with autism learn to cope with social situations “Over time, what you would expect to see is more normal social responding, being more interested in interacting with other people, more eye contact and more conversation,” said Kevin Pelphrey, of Yale University.

4 sentences, 140 tokens. 1/3 of this passage's questions were captured_correct -- only those are shown below, plus a fixed topic-probe question for comparison (no gold answer, so no f1).

| # | question | gold | pred | f1 | sent_entropy_norm | sent_pr_norm | tok_entropy_norm | tok_pr_norm |
|---|---|---|---|---|---|---|---|---|
| 1 | What difference did oxytocin make to the brains of autistic children, as shown in brain scans? | Improved responses to faces | improved brain responses to facial expressions | 0.60 | 0.822 | 0.626 | 0.620 | 0.048 |
| probe | What is the main topic of the passage? | -- | oxytocin could help children with autism | -- | 0.799 | 0.591 | 0.510 | 0.033 |

### Passage 4

**Passage:**

> Part of the reason for this is that air travel is dangerous so standards are much higher. “If you fly commercial airlines, they often say, ‘Oh, a small component has failed – we have to go back to the gate,’” Singh said. “And that’s an established industry with 60 years of legacy! I hate to think that a drone might come down on a busy road.” Part of the solution, Singh said, is planning for every situation: “If things fail, the vehicle has to do something reasonable.”

6 sentences, 119 tokens. 1/3 of this passage's questions were captured_correct -- only those are shown below, plus a fixed topic-probe question for comparison (no gold answer, so no f1).

| # | question | gold | pred | f1 | sent_entropy_norm | sent_pr_norm | tok_entropy_norm | tok_pr_norm |
|---|---|---|---|---|---|---|---|---|
| 1 | According to Singh, why do passenger airplanes often return to the gate? | A small part has failed | a small component has failed | 0.80 | 0.839 | 0.636 | 0.713 | 0.092 |
| probe | What is the main topic of the passage? | -- | planning for every situation | -- | 0.798 | 0.584 | 0.513 | 0.046 |

### Passage 5

**Passage:**

> Do you want your child to be good at sports, play for the school team and, maybe one day, even compete in international competitions? Well, try to make sure that your future Olympian or World Cup winner is born in November or October. A study has found that school pupils born in those months are fitter than everyone else in their class. November- and October-born children were fitter, stronger and more powerful than those born in the other ten months of the year, especially those whose birthdays were in April or June. Dr. Gavin Sandercock of Essex University found in his study that autumn-born children had “a clear physical advantage” over their classmates.

6 sentences, 140 tokens. 1/3 of this passage's questions were captured_correct -- only those are shown below, plus a fixed topic-probe question for comparison (no gold answer, so no f1).

| # | question | gold | pred | f1 | sent_entropy_norm | sent_pr_norm | tok_entropy_norm | tok_pr_norm |
|---|---|---|---|---|---|---|---|---|
| 1 | Who does the article suggest to be the weakest in comparison with children born in October and November? | Children born in April and June | November- and October-born children were fitter, stronger and more powerful than those born in the other ten months of the year, especially those whose birthdays were in April or June | 0.33 | 0.714 | 0.439 | 0.592 | 0.043 |
| probe | What is the main topic of the passage? | -- | try to make sure that your future Olympian or World Cup winner is born in November or October | -- | 0.598 | 0.284 | 0.460 | 0.020 |

### Passage 6

**Passage:**

> The ice-cream shop is in a documentary by film-makers Rob and Lisa Fruchtman. Sweet Dreams, which tells the story of how the women have made a promising post-genocide future, also includes the female drummers. The film has been shown in more than a dozen countries, including the US, UK and several African states. “We feel the film is about resilience, hope, bravery, resourcefulness and the ability to change the course of your own life,” says Lisa Fruchtman.

4 sentences, 108 tokens. 1/3 of this passage's questions were captured_correct -- only those are shown below, plus a fixed topic-probe question for comparison (no gold answer, so no f1).

| # | question | gold | pred | f1 | sent_entropy_norm | sent_pr_norm | tok_entropy_norm | tok_pr_norm |
|---|---|---|---|---|---|---|---|---|
| 1 | What is the name of Rob and Lisa Fruchtman’s film? | Sweet Dreams | Sweet Dreams | 1.00 | 0.953 | 0.864 | 0.600 | 0.052 |
| probe | What is the main topic of the passage? | -- | how the women have made a promising post-genocide future | -- | 0.888 | 0.742 | 0.519 | 0.039 |

Avg over 6 captured=Y question(s) -- sentence: entropy_norm=0.830, pr_norm=0.627, top1=0.351, top2=0.623, top3=0.861; token: entropy_norm=0.606, pr_norm=0.052, top5=0.682, top10=0.728, top15=0.761

## OneStopQA / HARD

6/6 captured (tried 13 candidates)

### Passage 1

**Passage:**

> South American Indians have chewed coca leaves for centuries. The leaves reputedly provide energy and are said to have medicinal qualities. Supporters of Bolivia’s position praised it for doing the right thing by defending the rights of indigenous people. “The Bolivian move is inspirational and groundbreaking,” said Danny Kushlick, Head of External Affairs at the Transform Drug Policy Foundation, which promotes drug liberalization. “It shows that any country that has had enough of the war on drugs can change the terms of its engagement with the UN conventions.”

6 sentences, 116 tokens. 1/3 of this passage's questions were captured_correct -- only those are shown below, plus a fixed topic-probe question for comparison (no gold answer, so no f1).

| # | question | gold | pred | f1 | sent_entropy_norm | sent_pr_norm | tok_entropy_norm | tok_pr_norm |
|---|---|---|---|---|---|---|---|---|
| 1 | Who is Danny Kushlick? | A member of the Transform Drug Policy Foundation | Head of External Affairs at the Transform Drug Policy Foundation | 0.67 | 0.679 | 0.420 | 0.574 | 0.046 |
| probe | What is the main topic of the passage? | -- | war on drugs | -- | 0.642 | 0.397 | 0.474 | 0.034 |

### Passage 2

**Passage:**

> n octopus has made a brazen escape from the National Aquarium in New Zealand by breaking out of its tank, slithering down a 50-meter drainpipe and disappearing into the sea. In scenes reminiscent of Finding Nemo, Inky – a common New Zealand octopus – made his dash for freedom after the lid of his tank was accidentally left slightly ajar. Staff believe that in the middle of the night, while the aquarium was deserted, Inky clambered to the top of his glass enclosure, down the side of the tank and traveled across the floor of the aquarium. Rob Yarrell, national manager of the National Aquarium of New Zealand in Napier, said: “Octopuses are famous escape artists. I don’t think he was unhappy with us, or lonely, as octopuses are solitary creatures. But, he is such a curious boy. He would want to know what’s happening on the outside. That’s just his personality.”

9 sentences, 204 tokens. 1/3 of this passage's questions were captured_correct -- only those are shown below, plus a fixed topic-probe question for comparison (no gold answer, so no f1).

| # | question | gold | pred | f1 | sent_entropy_norm | sent_pr_norm | tok_entropy_norm | tok_pr_norm |
|---|---|---|---|---|---|---|---|---|
| 1 | How does Yarrell describe Inky’s personality? | Curious | curious boy | 0.67 | 0.895 | 0.711 | 0.729 | 0.093 |
| probe | What is the main topic of the passage? | -- | outside | -- | 0.894 | 0.699 | 0.597 | 0.046 |

### Passage 3

**Passage:**

> From all across Rwanda, and even parts of neighboring Burundi, people flock to the southern town of Butare to a little shop called Inzozi Nziza (Sweet Dreams). They come for a taste of the unknown, something most have never tasted – the sweet, cold, velvety embrace of ice cream. Here, at the central African country’s first ice-cream parlor, customers can buy scoops in sweet cream, passion fruit, strawberry and pineapple flavors. Toppings include fresh fruit, honey, chocolate chips and granola. Black tea and coffee are also on sale.

5 sentences, 124 tokens. 1/3 of this passage's questions were captured_correct -- only those are shown below, plus a fixed topic-probe question for comparison (no gold answer, so no f1).

| # | question | gold | pred | f1 | sent_entropy_norm | sent_pr_norm | tok_entropy_norm | tok_pr_norm |
|---|---|---|---|---|---|---|---|---|
| 1 | Where is the ice cream shop located? | Butare | the southern town of Butare | 0.33 | 0.869 | 0.666 | 0.612 | 0.051 |
| probe | What is the main topic of the passage? | -- | a taste of the unknown | -- | 0.847 | 0.642 | 0.525 | 0.036 |

### Passage 4

**Passage:**

> But it isn’t just students who would benefit from a later start. Kelley says the working day should be more forgiving of our natural rhythms. Describing the average sleep loss per night for different age groups, he says: “Between 14 and 24, it’s more than two hours. For people aged between 24 and about 30 or 35, it’s about an hour and a half. That can continue up until you’re about 55 when it’s in balance again. The 10-year-old and 55-year-old wake and sleep naturally at the same time.”

7 sentences, 127 tokens. 1/3 of this passage's questions were captured_correct -- only those are shown below, plus a fixed topic-probe question for comparison (no gold answer, so no f1).

| # | question | gold | pred | f1 | sent_entropy_norm | sent_pr_norm | tok_entropy_norm | tok_pr_norm |
|---|---|---|---|---|---|---|---|---|
| 1 | What does Kelley say about the relationship between working hours and natural rhythms? | The working day should be adjusted to our natural rhythms | the working day should be more forgiving | 0.59 | 0.733 | 0.433 | 0.543 | 0.038 |
| probe | What is the main topic of the passage? | -- | the working day should be more forgiving of our natural rhythms | -- | 0.749 | 0.454 | 0.522 | 0.035 |

### Passage 5

**Passage:**

> Huber said about Amazon: “I have heard them say that many packages are lightweight – a drone can carry a kilogram for 15 minutes. If you have a vehicle that can go into a neighborhood, it can deliver from that base. You need a 15-minute distance and typical off-the-shelf drones have about that distance.” It’s one way, he said, of making sure the surrounding population is relatively safe. “The larger the distance the drone travels, the more dangerous it becomes.” Of course, safety remains a major concern – Singh points out that, for a commercial aircraft to be considered skyworthy, it has to prove a rate of one serious failure every one million hours. Drones, he said, are “one or two orders of magnitude away” from that benchmark. “The Reaper drone has one failure in 10,000 hours,” Singh said. An oil leak, by the way, doesn’t count as catastrophic failure – something has to fall out of the sky.

9 sentences, 216 tokens. 2/3 of this passage's questions were captured_correct -- only those are shown below, plus a fixed topic-probe question for comparison (no gold answer, so no f1).

| # | question | gold | pred | f1 | sent_entropy_norm | sent_pr_norm | tok_entropy_norm | tok_pr_norm |
|---|---|---|---|---|---|---|---|---|
| 1 | What happens as you increase the distance a drone travels to deliver a package? | The drone becomes more dangerous to people | the more dangerous | 0.60 | 0.917 | 0.750 | 0.737 | 0.077 |
| 2 | What determines if a passenger plane is allowed to operate? | The number of serious failures every one million hours | one serious failure every one million hours | 0.63 | 0.948 | 0.818 | 0.716 | 0.081 |
| probe | What is the main topic of the passage? | -- | safety | -- | 0.924 | 0.774 | 0.620 | 0.056 |

### Passage 6

**Passage:**

> The loans Duran swindled from banks were his way of regulating and denouncing this situation, he said. He started slowly. “I filled out a few credit applications with my real details. They denied me, but I just wanted to get a feel for what they were asking for.” From there, the former table-tennis coach began to weave an intricate web of accounts, payments and transfers. “I was learning constantly.” By the summer of 2007, he had discovered how to make the system work, applying for loans under the name of a false television production company. “Then, I managed to get a lot.” €492,000, to be exact.

9 sentences, 144 tokens. 1/3 of this passage's questions were captured_correct -- only those are shown below, plus a fixed topic-probe question for comparison (no gold answer, so no f1).

| # | question | gold | pred | f1 | sent_entropy_norm | sent_pr_norm | tok_entropy_norm | tok_pr_norm |
|---|---|---|---|---|---|---|---|---|
| 1 | How much money did Duran manage to take out in loans? | €492,000 in total | €492,000 | 0.50 | 0.925 | 0.758 | 0.682 | 0.092 |
| probe | What is the main topic of the passage? | -- | learning | -- | 0.926 | 0.763 | 0.681 | 0.096 |

Avg over 7 captured=Y question(s) -- sentence: entropy_norm=0.852, pr_norm=0.651, top1=0.269, top2=0.493, top3=0.686; token: entropy_norm=0.656, pr_norm=0.068, top5=0.585, top10=0.679, top15=0.718

## SQuAD / N/A

6/6 captured (tried 6 candidates)

### Passage 1

**Passage:**

> Although sizable Orthodox Jewish communities are located throughout the United States, many American Orthodox Jews live in New York State, particularly in the New York City Metropolitan Area. Two of the main Orthodox communities in the United States are located in New York City and Rockland County. In New York City, the neighborhoods of Borough Park, Midwood, Williamsburg, and Crown Heights, located in the borough of Brooklyn, have particularly large Orthodox communities. The most rapidly growing community of American Orthodox Jews is located in Rockland County and the Hudson Valley of New York, including the communities of Monsey, Monroe, New Square, and Kiryas Joel. There are also sizable and rapidly growing Orthodox communities throughout New Jersey, particularly in Lakewood, Teaneck, Englewood, Passaic, and Fair Lawn.

5 sentences, 161 tokens. 4/4 of this passage's questions were captured_correct -- only those are shown below, plus a fixed topic-probe question for comparison (no gold answer, so no f1).

| # | question | gold | pred | f1 | sent_entropy_norm | sent_pr_norm | tok_entropy_norm | tok_pr_norm |
|---|---|---|---|---|---|---|---|---|
| 1 | Borough Park, Midwood, Williamsburg and Crown heights have particularly large communities of what? | American Orthodox Jews | American Orthodox Jews | 1.00 | 0.990 | 0.957 | 0.561 | 0.047 |
| 2 | Where is a sizeable and rapidly growing Orthodox community currently located besides New York State? | New Jersey | New Jersey | 1.00 | 0.911 | 0.766 | 0.636 | 0.060 |
| 3 | Where is the most rapidly growing community of American orthodox jews located? | Rockland County | Rockland County | 1.00 | 0.856 | 0.682 | 0.523 | 0.041 |
| 4 | Where do many American Orthodox Jews live? | New York State | New York State | 1.00 | 0.847 | 0.635 | 0.642 | 0.050 |
| probe | What is the main topic of the passage? | -- | Orthodox | -- | 0.733 | 0.524 | 0.387 | 0.019 |

### Passage 2

**Passage:**

> These areas, quartiers sensibles ("sensitive quarters"), are in northern and eastern Paris, namely around its Goutte d'Or and Belleville neighbourhoods. To the north of the city they are grouped mainly in the Seine-Saint-Denis department, and to a lesser extreme to the east in the Val-d'Oise department. Other difficult areas are located in the Seine valley, in Évry et Corbeil-Essonnes (Essonne), in Mureaux, Mantes-la-Jolie (Yvelines), and scattered among social housing districts created by Delouvrier's 1961 "ville nouvelle" political initiative.

3 sentences, 141 tokens. 3/3 of this passage's questions were captured_correct -- only those are shown below, plus a fixed topic-probe question for comparison (no gold answer, so no f1).

| # | question | gold | pred | f1 | sent_entropy_norm | sent_pr_norm | tok_entropy_norm | tok_pr_norm |
|---|---|---|---|---|---|---|---|---|
| 1 | Where are the quartiers sensibles located? | northern and eastern Paris | northern and eastern Paris | 1.00 | 0.997 | 0.989 | 0.598 | 0.045 |
| 2 | What two neighborhoods are the centers of the quartiers sensibles? | Goutte d'Or and Belleville | Goutte d'Or and Belleville | 1.00 | 0.992 | 0.974 | 0.626 | 0.048 |
| 3 | Why were these neighborhoods created? | Delouvrier's 1961 "ville nouvelle" political initiative | Delouvrier's 1961 "ville nouvelle" political initiative | 1.00 | 0.843 | 0.635 | 0.603 | 0.039 |
| probe | What is the main topic of the passage? | -- | difficult | -- | 0.982 | 0.939 | 0.460 | 0.027 |

### Passage 3

**Passage:**

> When Emperor Kammu moved the capital to Heian-kyō (Kyōto), which remained the imperial capital for the next 1,000 years, he did so not only to strengthen imperial authority but also to improve his seat of government geopolitically. Nara was abandoned after only 70 years in part due to the ascendancy of Dōkyō and the encroaching secular power of the Buddhist institutions there. Kyōto had good river access to the sea and could be reached by land routes from the eastern provinces. The early Heian period (784–967) continued Nara culture; the Heian capital was patterned on the Chinese Tang capital at Chang'an, as was Nara, but on a larger scale than Nara. Kammu endeavoured to improve the Tang-style administrative system which was in use. Known as the ritsuryō, this system attempted to recreate the Tang imperium in Japan, despite the "tremendous differences in the levels of development between the two countries". Despite the decline of the Taika-Taihō reforms, imperial government was vigorous during the early Heian period. Indeed, Kammu's avoidance of drastic reform decreased the intensity of political struggles, and he became recognized as one of Japan's most forceful emperors.

8 sentences, 262 tokens. 5/5 of this passage's questions were captured_correct -- only those are shown below, plus a fixed topic-probe question for comparison (no gold answer, so no f1).

| # | question | gold | pred | f1 | sent_entropy_norm | sent_pr_norm | tok_entropy_norm | tok_pr_norm |
|---|---|---|---|---|---|---|---|---|
| 1 | Heian was Japan's capital for how many years? | 1,000 | 1,000 | 1.00 | 0.949 | 0.820 | 0.712 | 0.059 |
| 2 | Nara was the former capital for how many years? | 70 | 70 | 1.00 | 0.949 | 0.817 | 0.680 | 0.050 |
| 3 | What religion was gaining popularity in Nara? | Buddhist | Buddhist | 1.00 | 0.967 | 0.873 | 0.747 | 0.067 |
| 4 | What time period was the early Heian era? | 784–967 | 784–967 | 1.00 | 0.922 | 0.773 | 0.625 | 0.042 |
| 5 | Kanmu modeled his government after what Chinese capital? | Tang | Tang | 1.00 | 0.932 | 0.796 | 0.689 | 0.058 |
| probe | What is the main topic of the passage? | -- | improve his seat of government | -- | 0.938 | 0.808 | 0.667 | 0.047 |

### Passage 4

**Passage:**

> Switzerland has a dense network of cities, where large, medium and small cities are complementary. The plateau is very densely populated with about 450 people per km2 and the landscape continually shows signs of human presence. The weight of the largest metropolitan areas, which are Zürich, Geneva–Lausanne, Basel and Bern tend to increase. In international comparison the importance of these urban areas is stronger than their number of inhabitants suggests. In addition the two main centers of Zürich and Geneva are recognized for their particularly great quality of life.

5 sentences, 110 tokens. 3/3 of this passage's questions were captured_correct -- only those are shown below, plus a fixed topic-probe question for comparison (no gold answer, so no f1).

| # | question | gold | pred | f1 | sent_entropy_norm | sent_pr_norm | tok_entropy_norm | tok_pr_norm |
|---|---|---|---|---|---|---|---|---|
| 1 | What is the population density of the plateau? | 450 people per km2 | 450 people per km2 | 1.00 | 0.939 | 0.839 | 0.573 | 0.054 |
| 2 | Which 2 centers are recognized for their particularly great quality of life? | Zürich and Geneva | Zürich and Geneva | 1.00 | 0.952 | 0.808 | 0.594 | 0.079 |
| 3 | What does the weight of the largest metropolitan areas tend to do? | increase | increase | 1.00 | 0.995 | 0.979 | 0.564 | 0.063 |
| probe | What is the main topic of the passage? | -- | the landscape continually shows signs of human presence | -- | 0.900 | 0.781 | 0.444 | 0.038 |

### Passage 5

**Passage:**

> In 1937, IBM's tabulating equipment enabled organizations to process unprecedented amounts of data, its clients including the U.S. Government, during its first effort to maintain the employment records for 26 million people pursuant to the Social Security Act, and the Third Reich, largely through the German subsidiary Dehomag. During the Second World War the company produced small arms for the American war effort (M1 Carbine, and Browning Automatic Rifle). IBM provided translation services for the Nuremberg Trials. In 1947, IBM opened its first office in Bahrain, as well as an office in Saudi Arabia to service the needs of the Arabian-American Oil Company that would grow to become Saudi Business Machines (SBM).

6 sentences, 141 tokens. 5/5 of this passage's questions were captured_correct -- only those are shown below, plus a fixed topic-probe question for comparison (no gold answer, so no f1).

| # | question | gold | pred | f1 | sent_entropy_norm | sent_pr_norm | tok_entropy_norm | tok_pr_norm |
|---|---|---|---|---|---|---|---|---|
| 1 | What what was the name of the subsidiary working in Germany during World War 2? | Dehomag | Dehomag | 1.00 | 0.770 | 0.501 | 0.611 | 0.033 |
| 2 | Records for how many people were maintained by IBM in 1937? | 26 million | 26 million | 1.00 | 0.777 | 0.503 | 0.607 | 0.032 |
| 3 | What service did IBM provide for the Nuremberg Trials? | translation services | translation | 0.67 | 0.641 | 0.346 | 0.521 | 0.028 |
| 4 | What year did IBM open its first office in Bahrain? | 1947 | 1947 | 1.00 | 0.734 | 0.477 | 0.567 | 0.031 |
| 5 | What was the eventual name of the company that IBM operated in Saudi Arabia? | Saudi Business Machines | Saudi Business Machines | 1.00 | 0.754 | 0.481 | 0.556 | 0.028 |
| probe | What is the main topic of the passage? | -- | employment records for 26 million people pursuant to the Social Security Act | -- | 0.629 | 0.336 | 0.412 | 0.016 |

### Passage 6

**Passage:**

> The College Dropout was eventually issued by Roc-A-Fella in February 2004, shooting to number two on the Billboard 200 as his debut single, "Through the Wire" peaked at number fifteen on the Billboard Hot 100 chart for five weeks. "Slow Jamz", his second single featuring Twista and Jamie Foxx, became an even bigger success: it became the three musicians' first number one hit. The College Dropout received near-universal critical acclaim from contemporary music critics, was voted the top album of the year by two major music publications, and has consistently been ranked among the great hip-hop works and debut albums by artists. "Jesus Walks", the album's fourth single, perhaps exposed West to a wider audience; the song's subject matter concerns faith and Christianity. The song nevertheless reached the top 20 of the Billboard pop charts, despite industry executives' predictions that a song containing such blatant declarations of faith would never make it to radio. The College Dropout would eventually be certified triple platinum in the US, and garnered West 10 Grammy nominations, including Album of the Year, and Best Rap Album (which it received). During this period, West also founded GOOD Music, a record label and management company that would go on to house affiliate artists and producers, such as No I.D. and John Legend. At the time, the focal point of West's production style was the use of sped-up vocal samples from soul records. However, partly because of the acclaim of The College Dropout, such sampling had been much copied by others; with that overuse, and also because West felt he had become too dependent on the technique, he decided to find a new sound.

11 sentences, 341 tokens. 3/5 of this passage's questions were captured_correct -- only those are shown below, plus a fixed topic-probe question for comparison (no gold answer, so no f1).

| # | question | gold | pred | f1 | sent_entropy_norm | sent_pr_norm | tok_entropy_norm | tok_pr_norm |
|---|---|---|---|---|---|---|---|---|
| 1 | What was the name of the single off the debut album that gave Kanye mainstream attention? | Jesus Walks | Jesus Walks | 1.00 | 0.869 | 0.626 | 0.718 | 0.051 |
| 2 | What label did Kanye create following the success of his first album's release? | GOOD Music | GOOD Music | 1.00 | 0.893 | 0.690 | 0.732 | 0.057 |
| 3 | When was The College Dropout finally released? | February 2004 | February 2004 | 1.00 | 0.848 | 0.599 | 0.608 | 0.033 |
| probe | What is the main topic of the passage? | -- | faith and Christianity | -- | 0.848 | 0.584 | 0.693 | 0.046 |

Avg over 23 captured=Y question(s) -- sentence: entropy_norm=0.884, pr_norm=0.720, top1=0.302, top2=0.548, top3=0.734; token: entropy_norm=0.621, pr_norm=0.048, top5=0.641, top10=0.700, top15=0.735

