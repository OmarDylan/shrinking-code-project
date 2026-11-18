# Requirements — Car Yard

## Domain

Client: Used-car dealership / car buyer

Dataset: `auto93.csv`

The client wants an AI assistant that helps inspect, compare, explain, and select cars while reducing unnecessary test drives and avoiding poor or suspicious listings.

The requirements below are written in customer language. Each requirement is tagged with the project skill it demonstrates.

## Customer Requirements

| # | Customer Ask | Skill |
|---|---|---|
| 1 | From hundreds of cars, show me the unusually great buys. | remember |
| 2 | Show me what a typical good car, average car, and poor car look like in this inventory. | remember |
| 3 | Guess a car's fuel use before I see the final sticker information. | guess |
| 4 | Estimate how promising a newly listed car is when I only know some of its specifications. | guess |
| 5 | Tell me when one group of cars is really better than another instead of just looking better by chance. | certify |
| 6 | Tell me whether an apparent improvement in our recommendations is large enough to matter. | certify |
| 7 | Judge each new car as it arrives instead of waiting for the whole inventory to be collected. | flow |
| 8 | Warn me when the quality of newly arriving cars starts looking different from what we usually receive. | flow |
| 9 | Tell me why a poor car is slow, heavy, or thirsty. | blame |
| 10 | Tell me which specifications are most responsible for making a car a bad buy. | blame |
| 11 | Explain why you recommended one car instead of another car I was considering. | justify |
| 12 | Give me a short explanation I can show a skeptical buyer for why a car was accepted or rejected. | justify |
| 13 | Find me a great car while requiring as few test drives as possible. | choose |
| 14 | If my first choice is unavailable, find a few strong alternatives without inspecting the whole inventory. | choose |
| 15 | Tell me the smallest change that could turn a poor car into a more attractive option. | fix |
| 16 | Show me what would need to be different about this car for it to resemble the better cars in the inventory. | fix |
| 17 | Flag a strange listing that might contain a typo, bad measurement, or scam. | spot |
| 18 | Show me cars that are unusually different from the rest of the inventory even when they are not obviously bad. | spot |
| 19 | Tell me whether our car-selection process actually finds better cars than simply picking cars at random. | race |
| 20 | Tell me which evaluation approach finds good cars while requiring the fewest inspections or test drives. | race |

## Skill Coverage

Each of the ten project skills appears twice:

- remember: requirements 1–2
- guess: requirements 3–4
- certify: requirements 5–6
- flow: requirements 7–8
- blame: requirements 9–10
- justify: requirements 11–12
- choose: requirements 13–14
- fix: requirements 15–16
- spot: requirements 17–18
- race: requirements 19–20

These requirements define the client wish list before implementation. The implementation will be built incrementally so later skills can reuse structures and functions created by earlier skills.
