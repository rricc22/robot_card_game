# Robot Arm Card Battle Project Plan

This document outlines the steps to build a bimanual (two-arm) card playing robot using two SO-100 arms and the LeRobot framework. We are using a **modular approach**, separating the physical movement (LeRobot) from the game logic (Python + Computer Vision) to keep the project short and manageable.

## Project Goal
Two SO-100 arms play a basic "War" style card game (highest card wins). The system uses an overhead camera to read the cards and triggers the correct pre-trained arm movements.

---

## Part 1: Physical Setup & Hardware

1. **Card Modification (Crucial first step):**
   - Standard pincer grippers cannot pick up flat cards from a table.
   - *Solution:* 3D print small angled stands for the "deck" and the "center arena," or attach a small piece of sticky tack/tape to the cards to make them slightly elevated.
2. **Camera Placement:**
   - Mount a webcam looking top-down at the table arena. Ensure the lighting is consistent.
3. **Arm Positioning:**
   - Securely mount both SO-100 arms so they can reach their own "deck" and the shared "center arena" without colliding.

---

## Part 2: Train the Physical Skills (LeRobot)

We only need to train the specific, isolated movements. We do **not** train the game rules.

### Skill 1: `play_card` (Train for BOTH Arm A and Arm B)
- **Goal:** Pick up the top card from the deck and place it in the center arena.
- **Data Collection:** Teleoperate the arm to grab the top card, move it smoothly to the center, release it, and return to a neutral "Home" position.
- **Estimated Demos:** ~30-50 successful demonstrations per arm.

### Skill 2: `collect_winnings` (Train for BOTH Arm A and Arm B)
- **Goal:** Reach into the center arena, grab the winning cards (both yours and the opponent's), and pull them into a "score pile."
- **Data Collection:** Teleoperate the arm to move to the center, grab the cards, move them to the designated score area, release, and return Home.
- **Estimated Demos:** ~30-50 successful demonstrations per arm.

*Note on "Losing": We do not train a losing skill. If an arm loses, the Python script simply commands it to stay in the "Home" position.*

---

## Part 3: The "Brain" (Computer Vision & Logic)

This is a separate Python script that acts as the referee.

1. **Vision System (The Eyes):**
   - Write a script that takes a frame from the overhead webcam.
   - Use a simple OCR library (like `pytesseract` or `easyocr`) or a basic template matching script to read the numbers on the two cards placed in the center arena.
2. **Game Logic (The Referee):**
   - Compare the two numbers.
   - `if card_a > card_b:` -> Arm A wins.
   - `elif card_b > card_a:` -> Arm B wins.
   - `else:` -> Tie (Handle as desired, e.g., both arms do nothing).

---

## Part 4: Putting it Together (The Main Loop)

Create a main Python loop that orchestrates the game:

1. **Start Round:** Script triggers both LeRobot models to execute `play_card`.
2. **Wait:** Pause for 2-3 seconds to let the arms finish moving and clear the camera's view.
3. **Read:** Trigger the Vision System to read the cards in the center.
4. **Decide:** Execute Game Logic to determine the winner.
5. **Act:** 
   - If Arm A wins, script triggers Arm A's LeRobot model to execute `collect_winnings`. Arm B stays Home.
   - If Arm B wins, script triggers Arm B's LeRobot model to execute `collect_winnings`. Arm A stays Home.
6. **Repeat:** Loop back to Step 1 for the next round.
