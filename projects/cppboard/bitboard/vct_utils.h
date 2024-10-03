#ifndef VCT_UTILS
#define VCT_UTILS

#include <cstdlib>
#include <algorithm>
#include "utils.h"
#include "board.h"

enum BoardValue { UNKNOWN = 0, POSITIVE, NEGATIVE, NONE_VALUE };

class VCTBoard : public Board
{
    public:
        VCTBoard() { Initialize(); }
        VCTBoard(const Board& board) { memcpy(this, &board, sizeof(Board)); }
        VCTBoard(const VCTBoard& board) { memcpy(this, &board, sizeof(Board)); }
        BoardValue Evaluate(StoneType attacker, UC* actions) const;

};

inline
bool FilterReplicatedActions(UC* actions)
{
    if (!actions[0]) return false;
    int num = actions[0];
    std::sort(actions + 1, actions + 1 + num);
    actions[0] = 1;
    bool replicated = false;
    for (int i = 2; i <= num; i++)
        if (actions[i] == actions[actions[0]]) replicated = true;
        else actions[++actions[0]] = actions[i];
    return replicated;
}

inline
BoardValue VCTBoard::Evaluate(StoneType attacker, UC* actions) const
{
    const StoneType player = bit.GetPlayer();
    actions[0] = 0;
    if (player != attacker) 
    {
        if (GetActions(player, OPEN_FOUR, true) |
            GetActions(player, FOUR, true))                  return NEGATIVE;
        if (GetActions(player, OPEN_FOUR, false))            return POSITIVE;
        if (GetActions(player, FOUR, false, actions))        return UNKNOWN;
        if (GetActions(player, OPEN_THREE, true) ||
            !GetActions(player, OPEN_THREE, false, actions)) return NEGATIVE;
        GetActions(player, THREE, true, actions);            return UNKNOWN;
    }
    else
    {
        if (GetActions(player, OPEN_FOUR, true) |
            GetActions(player, FOUR, true))                 return POSITIVE;
        if (GetActions(player, OPEN_FOUR, false))           return NEGATIVE;
        if (GetActions(player, FOUR, false, actions))       return UNKNOWN;
        if (GetActions(player, OPEN_THREE, true))           return POSITIVE;
        if (GetActions(player, OPEN_THREE, false, actions)) 
        {
            GetActions(player, THREE, true, actions);
            FilterReplicatedActions(actions);
            return UNKNOWN;
        }
    }
    static int three = static_cast<int>(THREE);
    static int openTwo = static_cast<int>(OPEN_TWO);
    static UC** actionTable = LineShape::GetActionTable();
    static UC twoActions[STONE_NUM + 1], threeActions[STONE_NUM + 1];
    int ply = static_cast<int>(player);
    twoActions[0] = 0;
    threeActions[0] = 0;
    for (int row = 0; row < BOARD_SIZE; row++)
    {
        int threeIndex = GetShapeIndex(three, ply, 1, row);
        U32 threeCodedActions[4] = {
            codedActions[threeIndex] & CODING_MASK,   
            codedActions[threeIndex] >> CODING_LENGTH, 
            codedActions[threeIndex+1] & CODING_MASK, 
            codedActions[threeIndex+1] >> CODING_LENGTH
        };
        U32 threeOrActions = threeCodedActions[0] | threeCodedActions[1] |
                             threeCodedActions[2] | threeCodedActions[3];
        // to check if there is a three in open two & three
        static UC threeActionsAtRow[BOARD_SIZE + 1];
        if (threeOrActions)
        {
            U32 threeXorActions = threeCodedActions[0] ^ threeCodedActions[1] ^
                                  threeCodedActions[2] ^ threeCodedActions[3];
            U32 doubleThree = threeOrActions ^ threeXorActions;
            if (doubleThree)
            {
                UC* handle = actionTable[doubleThree];
                actions[++actions[0]] = ActionFlatten(row, handle[1]);
                return POSITIVE;
            }
            UC* handle = actionTable[threeOrActions];
            threeActionsAtRow[0] = 0;
            for (int i = 1; i <= handle[0]; i++)
            {
                threeActions[++threeActions[0]] = ActionFlatten(row, handle[i]);
                // to check if there is a three in open two & three
                threeActionsAtRow[++threeActionsAtRow[0]] = handle[i];
            }
        }

        int openTwoIndex = GetShapeIndex(openTwo, ply, 1, row);
        U32 openTwoCodedActions[4] = {
            codedActions[openTwoIndex] & CODING_MASK,   
            codedActions[openTwoIndex] >> CODING_LENGTH, 
            codedActions[openTwoIndex+1] & CODING_MASK, 
            codedActions[openTwoIndex+1] >> CODING_LENGTH
        };
        U32 openTwoOrActions = openTwoCodedActions[0] | openTwoCodedActions[1] |
                               openTwoCodedActions[2] | openTwoCodedActions[3];
        if (openTwoOrActions)
        {
            UC* handle = actionTable[openTwoOrActions];
            for (int i = 1; i <= handle[0]; i++)
                twoActions[++twoActions[0]] = ActionFlatten(row, handle[i]);
        }

        U32 threeOrTwo = threeOrActions | openTwoOrActions;
        if (!threeOrTwo) continue;
        U32 threeXorTwo = threeOrActions ^ openTwoOrActions;
        U32 threeTwo = threeOrTwo ^ threeXorTwo;
        if (!threeTwo) continue;
        U32 threeTwoCodedActions[4] = {
            threeCodedActions[0] ^ openTwoCodedActions[0], 
            threeCodedActions[1] ^ openTwoCodedActions[1], 
            threeCodedActions[2] ^ openTwoCodedActions[2], 
            threeCodedActions[3] ^ openTwoCodedActions[3]
        };
        threeOrTwo = threeTwoCodedActions[0] | threeTwoCodedActions[1] |
                     threeTwoCodedActions[2] | threeTwoCodedActions[3];
        threeXorTwo = threeTwoCodedActions[0] ^ threeTwoCodedActions[1] ^
                      threeTwoCodedActions[2] ^ threeTwoCodedActions[3];
        threeTwo = threeOrTwo ^ threeXorTwo;
        if (!threeTwo) continue;
        UC* handle = actionTable[threeTwo];
        for (int i = 1; i <= handle[0]; i++)
        {
            UC col = handle[i];
            // to check if there is a three in open two & three
            for (int j = 1; j <= threeActionsAtRow[0]; j++)
                if (col == threeActionsAtRow[j])
                {
                    UC action = ActionFlatten(row, col);
                    // to check if the three is effective
                    static UC tempActions[STONE_NUM + 1];
                    VCTBoard board(*this);
                    board.Move(action);
                    board.Evaluate(attacker, tempActions);
                    board.Move(tempActions[1]);
                    if (POSITIVE != board.Evaluate(attacker, tempActions)) continue;
                    actions[++actions[0]] = action;
                    return POSITIVE;
                }
        }
    }
    for (int i = 1; i <= twoActions[0]; i++) 
        actions[++actions[0]] = twoActions[i];
    for (int i = 1; i <= threeActions[0]; i++) 
        actions[++actions[0]] = threeActions[i];
    if (actions[0]) return UNKNOWN;
    else return NEGATIVE;
}

#endif