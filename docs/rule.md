# Supported Mahjong Games

Mahjax currently implements 4-player Japanese Riichi Mahjong variants.
To get familiar with the basic rules, please refer to the official rulebook:
[European Riichi Mahjong Rules (2016, EN)](http://mahjong-europe.org/portal/images/docs/Riichi-rules-2016-EN.pdf).

While different variants require different strategies due to their specific rules, we believe the core skills they demand are the same:

- Efficient tile and hand evaluation (reasonable combination calculation)
- Risk management (defense vs. attack)
- Reading and inferring opponents’ hands

At the moment, Mahjax provides two rule variants:

- **No red (aka no red dora)**
- **Red (with red dora)**

The “red” in red mahjong refers to **additional dora tiles assigned to certain 5s** (usually 5m, 5p, 5s). This has a large impact on strategy:

- In **no red** mahjong, players must rely more on *yaku* (hand patterns) to increase score.
- In **red** mahjong, players prioritize **efficiency** to complete a winning hand quickly and must pay closer attention to **risk**, as the average hand value is higher and the chance of dealing into a high-value RON increases.

---

## No Red Mahjong

Although no red dora rules are not as popular in the gaming industry or in existing Mahjong AI research, they are widely played in Japan. Some professional leagues adopting this style include:

- [Nihon Pro Mahjong (NIHON PRO MAHJONG)](https://npm2001.com/about/rule/)
- [Saikouisen Nihon Pro Mahjong Kyoukai (SAIKOUISEN)](https://saikouisen.com/about/rules/)

These rules emphasize careful yaku construction and more traditional scoring without the volatility introduced by red dora.

---

## Red Mahjong

Red mahjong, especially the rule set used by [Tenhou](https://tenhou.net/), is currently the **most popular** variant both in AI research and in commercial online games.

### Research Projects Using/Targeting Red Rules

- suphnx
- mjx
- mortal
- naga

### Games Using/Supporting Red Rules

- Tenhou
- Mahjong Soul
- MJ Mahjong

These environments typically feature fast-paced play and high average hand values, making them attractive both for players and for AI research.

---

## Future Pathways

Planned future extensions for Mahjax include:

- Supporting additional **4-player Japanese Riichi Mahjong** rule sets
- Integrating **3-player Japanese Mahjong** rules

These expansions aim to make Mahjax a more comprehensive platform for both Mahjong AI research and gameplay experimentation.
