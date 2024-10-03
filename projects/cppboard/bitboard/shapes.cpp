#include "shapes.h"

U32* GetWhiteToBlackTable();

U32* Shape::whiteToBlackTable = GetWhiteToBlackTable();

inline StoneType GetLineItem(U32 line, int pos)
{
    line >>= 2 * pos;
    return static_cast<StoneType>(MASK & (~line));
}

U32* GetWhiteToBlackTable()
{
    U32* ptr = new U32[MAP_SIZE];
    for (int i = 0; i < MAP_SIZE; i++)
    {
        U32 whiteLine = i, blackLine = 0;
        for (int j = 0; j < BOARD_SIZE; j++)
        {
            StoneType stone = GetLineItem(whiteLine, j);
            if (NONE == stone) break;
            switch (stone)
            {
            case BLACK:
                stone = WHITE;
                break;
            case WHITE:
                stone = BLACK;
                break;
            default:
                break;
            }
            blackLine ^= (MASK & (~static_cast<U32>(stone))) << (2 * j);
        }
        ptr[i] = blackLine;
    }
    return ptr;
}

LineShape* Shape::MetaFindAllShapes(U32 line)
{   
    LineShape* ptr = new LineShape(line);
    StoneType player = BLACK;
    bool flag = false;
    int shapeStart = static_cast<int>(OPEN_FOUR);
    int shapeEnd = static_cast<int>(OPEN_TWO);
    int isAttackerStart = static_cast<int>(false);
    int isAttackerEnd = static_cast<int>(true);
    bool isFive = Check(line, player, FIVE, true);
    ptr->SetFive(isFive);
    if (isFive) flag = true;
    for (int sp = shapeStart; sp <= shapeEnd; sp++)
    {
        ShapeType shape = static_cast<ShapeType>(sp), targetShape;
        int actionNumMin = 1, actionNumMax = BOARD_SIZE;
        switch (shape)
        {
        case OPEN_FOUR:
            targetShape = FIVE;
            actionNumMin = 2;
            break;
        case FOUR:
            targetShape = FIVE;
            actionNumMax = 1;
            break;
        case OPEN_THREE:
            targetShape = OPEN_FOUR;
            break;
        case THREE:
            targetShape = FOUR;
            break;
        case OPEN_TWO:
            targetShape = OPEN_THREE;
            break;
        default:
            break;
        }
        for (int it = isAttackerStart; it <= isAttackerEnd; it++)
        {
            bool isAttacker = static_cast<bool>(it);
            U32 actionCoding = MetaFind(line, targetShape, isAttacker);
            if (!actionCoding) continue;
            UC actionNum = LineShape::GetActionTable()[actionCoding][0];
            if (actionNum >= actionNumMin && actionNum <= actionNumMax)
            {
                ptr->SetActions(shape, isAttacker, actionCoding);
                flag = true;
            }
        }
    }
    if (flag) return ptr;
    delete ptr;
    return LineShapeTable::End();
}

U32 Shape::MetaFind(U32 line, ShapeType targetShape, bool isAttacker)
{
    U32 actionCoding = 0;
    StoneType player = BLACK;
    StoneType opponent = WHITE;
    StoneType attacker = isAttacker ? player : opponent;
	if (Check(line, attacker, targetShape, true)) return actionCoding;

    UC emptyActions[BOARD_SIZE + 1] = { 0 };
	for (int pos = 0; pos < BOARD_SIZE; pos++)
	{
		StoneType item = GetLineItem(line, pos);
		if (NONE == item) break;
        if (EMPTY == item) emptyActions[++emptyActions[0]] = pos;
	}
	if (!emptyActions[0]) return actionCoding;

	for (int i = 1; i <= emptyActions[0]; i++)
	{
		UC act = emptyActions[i];
		if (Find(line^Bit::GetStone(attacker, act), 
                 attacker, targetShape, true)) 
            actionCoding ^= 1 << act;
	}
	if (isAttacker || (!actionCoding)) return actionCoding;

	StoneType defender = player;
	UC* attackerActions = LineShape::GetActionTable()[actionCoding];
    actionCoding = 0;
	for (int i = 1; i <= emptyActions[0]; i++)
	{
		UC defAct = emptyActions[i];
		U32 defenderLine = line ^ Bit::GetStone(defender, defAct);
		bool flag = true;
		for (int j = 1; j <= attackerActions[0]; j++)
		{
			UC attAct = attackerActions[j];
			if (defAct == attAct) continue;
			if (Find(defenderLine^Bit::GetStone(attacker, attAct),
				     attacker, targetShape, true))
			{
				flag = false;
				break;
			}
		}
		if (flag) actionCoding ^= 1 << defAct;
	}
    return actionCoding;
}
                      
