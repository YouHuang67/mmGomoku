#ifndef BOARD
#define BOARD

#include <iostream>
#include <algorithm>
#include "utils.h"
#include "board_bits.h"
#include "shapes.h"
#include "init.h"

const unsigned int ROW_LENGTH = 4;
const unsigned int ROW_MASK = (1 << ROW_LENGTH) - 1;
const unsigned int ANCHOR_NUM = (2 * BOARD_SIZE - 1) * 2 + BOARD_SIZE * 2;

UC ActionToAnchor(int row, int col, int dir);

inline UC ActionToAnchor(int row, int col, Direction dir)
    { return ActionToAnchor(row, col, static_cast<int>(dir)); }

inline UC ActionToAnchor(UC act, int dir)
    { return ActionToAnchor(act >> ROW_LENGTH, act & ROW_MASK, dir); }

inline UC ActionToAnchor(UC act, Direction dir)
    { return ActionToAnchor(act, static_cast<int>(dir)); }

class Board
{
    public:
        Board() { Initialize(); };
        Board(const Board& board) { (*this) = board; }
        Board& operator=(const Board& board);
        inline StoneType Winner() const { return winner; }
        inline U64 Key() const { return zobristKey; }
        inline bool IsOver() const { return winner != NONE; }
        void Move(UC act, StoneType ply = EMPTY);
        void Move(UC* acts, StoneType* plys = nullptr);
        void Undo(int step = 1);
        void UndoAndMove(int step, UC* acts, StoneType* plys = nullptr);
        bool GetActions(StoneType player, ShapeType shape, 
                        bool isAttacker, UC* actions = nullptr) const;
        inline StoneType GetPlayer() const { return bit.GetPlayer(); }
        inline int GetStep() const { return bit.GetStep(); }
        inline static U64 UpdateZobristKey(U64 key, StoneType ply, UC act)
            { return key ^ zobristTable[static_cast<int>(ply)][act]; }
        inline U32 GetLine(UC act, int dir) const
            { return bit.GetLine(act, static_cast<Direction>(dir)); }

    protected:
        static LineShapeInitialization* lsinit;
        static const int playerStart,     playerEnd;
        static const int shapeStart,      shapeEnd;
        static const int isAttackerStart, isAttackerEnd;
        static const int directionStart,  directionEnd;
        static int** anchorIndexTable;
        static U64** zobristTable;
        static U32*** rotatedActionCodingTable;
        // (anchor << 2) ^ direction
        static U32 ancsAndDirs[8 * STONE_NUM + 1];
        static const LineShape* lineshapesBefore[2][ANCHOR_NUM];
        static int GetShapeIndex(int sp, int ply, int it, int row);
        static void GetRotatedActionCodingTable();
        static void ActionsToAnchors(UC* acts);
        Bit bit;
        // actions[shapeIdx & attakerIdx][playerIdx][rowIdx][directionIdx/2]
        U32 codedActions[8*2*16*2];
        StoneType winner;
        const LineShape* lineshapes[2][ANCHOR_NUM];
        U64 zobristKey = 0;
        void Initialize();
        void SetEncodedActions(U32 eAct, int row, int dir, 
                               int sp, int ply, int it);
        void SetEncodedActionsByShape(const LineShape* lsBefore, 
                                      const LineShape* lsAfter, 
                                      UC act, int dir, int sp, 
                                      int ply, int it, int index);
        void UpdateLinesBefore();
        void UpdateLinesAfter();
        void CheckWinner();
        void UpdateKey();
        StoneType* MetaMove(UC* acts, StoneType* plys = nullptr);
        void MetaUndo(UC* acts, StoneType* plys, int step);
        U32 GetEncodedActions(int sp, int ply, int it, int row) const;
        friend std::ostream& operator<<(std::ostream& os, const Board& board);

};

inline
std::ostream& operator<<(std::ostream& os, const Board& board)
{
    os << board.bit;
    return os;
}

inline
void Board::Move(UC act, StoneType ply)
{
    ply = EMPTY == ply ? bit.GetPlayer() : ply;
    zobristKey = UpdateZobristKey(zobristKey, ply, act);
    bit.Move(act, ply);
    ancsAndDirs[0] = 0;
    for (int dir = directionStart; dir <= directionEnd; dir++)
            ancsAndDirs[++ancsAndDirs[0]] = 
                (static_cast<U32>(ActionToAnchor(act, dir)) << 2) ^ dir;
    UpdateLinesBefore();
    UpdateLinesAfter();
    CheckWinner();
}

inline
void Board::Move(UC* acts, StoneType* plys)
{
    MetaMove(acts, plys);
    ActionsToAnchors(acts);
    UpdateLinesBefore();
    UpdateLinesAfter();
    CheckWinner();
}

inline
void Board::Undo(int step)
{
    static UC acts[STONE_NUM + 1];
    static StoneType plys[STONE_NUM];
    MetaUndo(acts, plys, step);
    ActionsToAnchors(acts);
    UpdateLinesBefore();
    UpdateLinesAfter();
    CheckWinner();
}

inline
void Board::UndoAndMove(int step, UC* acts, StoneType* plys)
{
    static UC totalActs[2 * STONE_NUM + 1];
    static StoneType totalPlys[2 * STONE_NUM];
    MetaUndo(totalActs, totalPlys, step);
    plys = MetaMove(acts, plys);
    for (int i = 1; i <= acts[0]; i++) 
        if (totalPlys[totalActs[0] - 1] == plys[i - 1] && 
            totalActs[totalActs[0]] == acts[i] && totalActs[0]) totalActs[0]--;
        else totalActs[++totalActs[0]] = acts[i];
    ActionsToAnchors(totalActs);
    UpdateLinesBefore();
    UpdateLinesAfter();
    CheckWinner();
}

inline
bool Board::GetActions(StoneType player, ShapeType shape, 
                       bool isAttacker, UC* actions) const 
{
    if (!isAttacker && (FOUR == shape || OPEN_FOUR == shape))
    {
        player = BLACK == player ? WHITE : BLACK;
        isAttacker = true;
    }
    int ply = static_cast<int>(player);
    int sp  = static_cast<int>(shape);
    int it  = static_cast<int>(isAttacker);
    static UC** actionTable = LineShape::GetActionTable();
    U32 actionCoding = 0;
    if (!actions)
    {
        for (int row = 0; row < BOARD_SIZE; row++)
            actionCoding |= GetEncodedActions(sp, ply, it, row);
        return actionCoding > 0;
    }
    UC initNum = actions[0];
    for (int row = 0; row < BOARD_SIZE; row++)
    {
        actionCoding = GetEncodedActions(sp, ply, it, row);
        if (!actionCoding) continue;
        UC* handle = actionTable[actionCoding];
        for (int i = 1; i <= handle[0]; i++)
            actions[++actions[0]] = ActionFlatten(row, handle[i]);
    }
    return actions[0] > initNum;
}

inline
void Board::UpdateLinesBefore()
{
    for (int i = 1; i <= ancsAndDirs[0]; i++)
    {
        int dir = ancsAndDirs[i] & 3;
        UC anchor = static_cast<UC>(ancsAndDirs[i] >> 2);
        int index = anchorIndexTable[dir][anchor];
        lineshapesBefore[0][index] = lineshapes[0][index];
        lineshapesBefore[1][index] = lineshapes[1][index];
    }
}

inline
void Board::UpdateLinesAfter()
{
    static Shape shape;
    for (int i = 1; i <= ancsAndDirs[0]; i++)
    {
        int dir = ancsAndDirs[i] & 3;
        UC anchor = static_cast<UC>(ancsAndDirs[i] >> 2);
        U32 line = bit.GetLine(anchor, static_cast<Direction>(dir));
        const LineShape* lineshapesAfter[2] = { 
            shape.FindAllShapes(line), 
            shape.FindAllShapes(Shape::WhiteToBlack(line))
        };
        int index = anchorIndexTable[dir][anchor];
        for (int p = 0; p <= 1; p++)
        {
            const LineShape* lsBefore = lineshapesBefore[p][index];
            const LineShape* lsAfter  = lineshapesAfter[p];
            for (int sp = shapeStart; sp <= shapeEnd; sp++)
                SetEncodedActionsByShape(lsBefore, lsAfter, anchor, dir, sp, p, 1, index);
            SetEncodedActionsByShape(lsBefore, lsAfter, anchor, dir, 
                                     static_cast<int>(OPEN_THREE), p, 0, index);
            lineshapes[p][index] = lineshapesAfter[p];
        }
    }
}

inline
int Board::GetShapeIndex(int sp, int ply, int it, int row)
{
    int index;
    // all attaker actions are required to store
    // only the shape of open three for defender is needed to store
    if (it) index = sp - shapeStart;
    else index = shapeEnd - shapeStart + 1;
    index = (index << 1) ^ ply;
    index = (index << 4) ^ row;
    return index << 1;
}

inline
void Board::SetEncodedActions(U32 eAct, int row, int dir, 
                              int sp, int ply, int it)
{
    if ((1 & dir)) eAct <<= CODING_LENGTH;
    codedActions[GetShapeIndex(sp, ply, it, row) + (dir >> 1)] ^= eAct;
}

inline
void Board::SetEncodedActionsByShape(const LineShape* lsBefore, 
                                     const LineShape* lsAfter, 
                                     UC act, int dir, int sp, 
                                     int ply, int it, int index)
{
    ShapeType st = static_cast<ShapeType>(sp);
    U32 actionCodingBefore = 0;
    U32 actionCodingAfter  = 0;
    if (lsBefore) actionCodingBefore = lsBefore->GetActionCoding(st, it);
    if (lsAfter)  actionCodingAfter  = lsAfter->GetActionCoding(st, it);
    // find different actions
    U32 diff = actionCodingBefore ^ actionCodingAfter;
    if (!diff) return;
    if (ROW == static_cast<Direction>(dir))
    {
        SetEncodedActions(diff, act>>ROW_LENGTH, dir, sp, ply, it);
        return;
    }
    U32* actionCodings = rotatedActionCodingTable[index][diff];
    for (int i = 1; i <= actionCodings[0]; i++)
        SetEncodedActions(actionCodings[i]>>ROW_LENGTH, 
                          actionCodings[i]&ROW_MASK, dir, sp, ply, it);
}

inline
void Board::CheckWinner()
{
    StoneType player = bit.GetPreviousPlayer();
    int ply = static_cast<int>(player);
    winner = NONE;
    for (int i = 1; i <= ancsAndDirs[0]; i++)
    {
        int dir = ancsAndDirs[i] & 3;
        UC anchor = static_cast<UC>(ancsAndDirs[i] >> 2);
        int index = anchorIndexTable[dir][anchor];
        if (lineshapes[ply][index] && lineshapes[ply][index]->IsFive())
        {
            winner = player;
            return;
        }
    }
    if (STONE_NUM == bit.GetStep()) winner = EMPTY;
}

inline
void Board::ActionsToAnchors(UC* acts)
{
    ancsAndDirs[0] = 0;
    for (int i = 1; i <= acts[0]; i++)
    {
        UC act = acts[i];
        for (int dir = directionStart; dir <= directionEnd; dir++)
            ancsAndDirs[++ancsAndDirs[0]] = 
                (static_cast<U32>(ActionToAnchor(act, dir)) << 2) ^ dir;
    }
    int num = ancsAndDirs[0];
    std::sort(ancsAndDirs + 1, ancsAndDirs + 1 + num);
    ancsAndDirs[0] = 1;
    for (int i = 2; i <= num; i++)
        if (ancsAndDirs[ancsAndDirs[0]] != ancsAndDirs[i])
            ancsAndDirs[++ancsAndDirs[0]] = ancsAndDirs[i];
}

inline
StoneType* Board::MetaMove(UC* acts, StoneType* plys)
{
    static StoneType tPlys[STONE_NUM];
    for (int i = 1; i <= acts[0]; i++)
    {
        UC act = acts[i];
        StoneType ply;
        if (plys) ply = plys[i - 1];
        else 
        {
            ply = bit.GetPlayer();
            tPlys[i - 1] = ply;
        }
        zobristKey = UpdateZobristKey(zobristKey, ply, act);
        bit.Move(act, ply);
    }
    if (plys) return plys;
    return tPlys;
}

inline
void Board::MetaUndo(UC* acts, StoneType* plys, int step)
{
    acts[0] = 0;
    for (int i = 0; i < step; i++)
    {   
        StoneType ply = bit.GetPreviousPlayer();
        UC act = bit.Undo();
        zobristKey = UpdateZobristKey(zobristKey, ply, act);
        plys[acts[0]] = ply;
        acts[++acts[0]] = act;
    }
}

inline
U32 Board::GetEncodedActions(int sp, int ply, int it, int row) const
{
    int index = GetShapeIndex(sp, ply, it, row);
    return CODING_MASK & (
        codedActions[index] | 
       (codedActions[index] >> CODING_LENGTH) |
        codedActions[index+1] |
       (codedActions[index+1] >> CODING_LENGTH));
}

inline
UC ActionToAnchor(int row, int col, int dir)
{
    UC anchor;
    int colGap;
    switch (static_cast<Direction>(dir))
    {
    case ROW:
        anchor = ActionFlatten(row, 0);
        break;
    case COL:
        anchor = ActionFlatten(0, col);
        break;
    case DIA:
        if (row >= col) anchor = ActionFlatten(row - col, 0);
        else            anchor = ActionFlatten(0, col - row);
        break;
    case COU:
        colGap = BOARD_SIZE - 1 - col;
        if (row >= colGap) anchor = ActionFlatten(row - colGap, BOARD_SIZE - 1);
        else               anchor = ActionFlatten(0, row + col);
        break;
    default:
        break;
    }
    return anchor;
}

#endif