# Interpretability Summary (LIME vs SHAP)

Top 15 words pushing toward each class, aggregated across 20 explained examples per combo.
Cross-method agreement (words in BOTH lists) signals a robust learned cue.

## roberta_sarc

### Top sarcasm pushers

| Rank | LIME word | LIME weight | SHAP word | SHAP weight |
|---|---|---|---|---|
| 1 | for | 0.5825 | **forgot** | 0.4865 |
| 2 | **forgot** | 0.5146 | sure | 0.4034 |
| 3 | **noob** | 0.4861 | right | 0.3109 |
| 4 | just | 0.4436 | because | 0.2790 |
| 5 | about | 0.3873 | windows | 0.2723 |
| 6 | upvotat | 0.3510 | thanks | 0.2602 |
| 7 | using | 0.3295 | **that** | 0.2191 |
| 8 | **you** | 0.3144 | **you** | 0.2124 |
| 9 | t | 0.3075 | **noob** | 0.1973 |
| 10 | firefox | 0.3010 | same | 0.1748 |
| 11 | voi | 0.2938 | **but** | 0.1686 |
| 12 | serves | 0.2747 | hated | 0.1635 |
| 13 | **that** | 0.2300 | fifa | 0.1611 |
| 14 | **but** | 0.2289 | restul | 0.1534 |
| 15 | the | 0.2197 | good | 0.1489 |

**Cross-method agreement (sarcasm):** 5 / 15 top words appear in both lists: `but, forgot, noob, that, you`

### Top not-sarcasm pushers

| Rank | LIME word | LIME weight | SHAP word | SHAP weight |
|---|---|---|---|---|
| 1 | **you** | 0.2206 | of | 0.6195 |
| 2 | t | 0.0868 | **you** | 0.4569 |
| 3 | said | 0.0657 | doesn' | 0.4241 |
| 4 | didn | 0.0637 | some | 0.4073 |
| 5 | a | 0.0606 | voi | 0.3496 |
| 6 | there | 0.0606 | **upvoted** | 0.3273 |
| 7 | real | 0.0500 | porn | 0.3032 |
| 8 | automatically | 0.0443 | **home** | 0.2918 |
| 9 | **home** | 0.0417 | rules | 0.2411 |
| 10 | **am** | 0.0393 | **am** | 0.2317 |
| 11 | i | 0.0294 | in | 0.2287 |
| 12 | **upvoted** | 0.0284 | still | 0.2247 |
| 13 | with | 0.0233 | food | 0.2234 |
| 14 | lower | 0.0161 | house | 0.2120 |
| 15 | lash | 0.0134 | poi, | 0.1987 |

**Cross-method agreement (not-sarcasm):** 4 / 15 top words: `am, home, upvoted, you`

## roberta_twitter_no_hashtags

### Top sarcasm pushers

| Rank | LIME word | LIME weight | SHAP word | SHAP weight |
|---|---|---|---|---|
| 1 | **love** | 0.4801 | **love** | 0.5901 |
| 2 | another | 0.4025 | **way** | 0.5689 |
| 3 | **like** | 0.3253 | **decent** | 0.4929 |
| 4 | **no** | 0.3203 | hum | 0.4584 |
| 5 | the | 0.2856 | you | 0.2675 |
| 6 | to | 0.2676 | **no** | 0.2421 |
| 7 | **true** | 0.2064 | things." | 0.2354 |
| 8 | **way** | 0.1963 | just | 0.2236 |
| 9 | **decent** | 0.1961 | disappointed | 0.2189 |
| 10 | one | 0.1930 | **boring** | 0.2120 |
| 11 | **boring** | 0.1922 | **like** | 0.1999 |
| 12 | being | 0.1743 | not | 0.1971 |
| 13 | **quite** | 0.1649 | please | 0.1915 |
| 14 | i | 0.1580 | **true** | 0.1871 |
| 15 | call | 0.1570 | **quite** | 0.1733 |

**Cross-method agreement (sarcasm):** 8 / 15 top words appear in both lists: `boring, decent, like, love, no, quite, true, way`

### Top not-sarcasm pushers

| Rank | LIME word | LIME weight | SHAP word | SHAP weight |
|---|---|---|---|---|
| 1 | **are** | 0.4601 | **the** | 0.5402 |
| 2 | survey | 0.1935 | **are** | 0.4239 |
| 3 | **the** | 0.1359 | walk | 0.2617 |
| 4 | blogger | 0.1264 | for | 0.2245 |
| 5 | **sparkling_heart** | 0.1261 | **to** | 0.2044 |
| 6 | choose | 0.1231 | **sparkling_heart** | 0.1846 |
| 7 | elsewhere | 0.1078 | posts... | 0.1714 |
| 8 | drum | 0.0977 | away. | 0.1633 |
| 9 | **to** | 0.0846 | **of** | 0.1521 |
| 10 | fill | 0.0797 | amp; | 0.1422 |
| 11 | party | 0.0672 | accessories | 0.1241 |
| 12 | 5 | 0.0524 | are ' | 0.1230 |
| 13 | will | 0.0514 | over- | 0.1211 |
| 14 | **of** | 0.0397 | branding | 0.1151 |
| 15 | oliver | 0.0396 | by | 0.1116 |

**Cross-method agreement (not-sarcasm):** 5 / 15 top words: `are, of, sparkling_heart, the, to`
