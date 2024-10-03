#include "board.h"

int** GetAnchorIndexTable();
U64** GetZobristTable();

LineShapeInitialization* Board::lsinit = nullptr;
const int Board::playerStart     = static_cast<int>(BLACK);
const int Board::playerEnd       = static_cast<int>(WHITE);
const int Board::shapeStart      = static_cast<int>(OPEN_FOUR);
const int Board::shapeEnd        = static_cast<int>(OPEN_TWO);
const int Board::isAttackerStart = static_cast<int>(false);
const int Board::isAttackerEnd   = static_cast<int>(true);
const int Board::directionStart  = static_cast<int>(ROW);
const int Board::directionEnd    = static_cast<int>(COU);
int** Board::anchorIndexTable = GetAnchorIndexTable();
U64** Board::zobristTable = GetZobristTable();
U32*** Board::rotatedActionCodingTable = nullptr;
U32 Board::ancsAndDirs[8 * STONE_NUM + 1];
const LineShape* Board::lineshapesBefore[2][ANCHOR_NUM];


void Board::Initialize()
{ 
    if (!lsinit)
    {
        lsinit = new LineShapeInitialization(0);
        GetRotatedActionCodingTable();
        memset(lineshapesBefore[0], 0, sizeof(LineShape*) * ANCHOR_NUM);
        memset(lineshapesBefore[1], 0, sizeof(LineShape*) * ANCHOR_NUM);
    }
    winner = NONE;
    memset(codedActions, 0, sizeof(codedActions));
    memset(lineshapes[0], 0, sizeof(lineshapes[0]));
    memset(lineshapes[1], 0, sizeof(lineshapes[1]));
}

Board& Board::operator=(const Board& board) 
{ 
    memcpy(this, &board, sizeof(Board)); 
    return *this;
}

int** GetAnchorIndexTable()
{
    int** ptr = new int*[4];
    int count = 0;
    for (int dir = 0; dir < 4; dir++)
    {
        ptr[dir] = new int[1 << 8];
        for (int i = 0; i < STONE_NUM; i++)
        {
            int row = i / BOARD_SIZE, col = i % BOARD_SIZE;
            UC act = static_cast<UC>(row << 4) ^ 
                     static_cast<UC>(col);
            const UC anchor = ActionToAnchor(act, dir);
            if (anchor == act) ptr[dir][act] = count++;
            else ptr[dir][act] = ptr[dir][anchor];
        }
    }
    return ptr;
}

U64** GetZobristTable()
{
    srand(1 << 10);
    U64** ptr = new U64*[2];
    ptr[0] = InitializeZobristKeys(1 << 8);
    ptr[1] = InitializeZobristKeys(1 << 8);
    return ptr;
}

inline
U32 CodeAction(int row, int col)
{
    return static_cast<U32>(((1 << col) << ROW_LENGTH) ^ row);
}

void Board::GetRotatedActionCodingTable()
{
    // tablePtr[anchorIndex][actionCoding][rowActionCoding]
    rotatedActionCodingTable = new U32**[ANCHOR_NUM];
    UC** actionTable = LineShape::GetActionTable();
    for (int d = directionStart + 1; d <= directionEnd; d++)
    {
        Direction dir = static_cast<Direction>(d);
        int dx, dy;
        switch (dir)
        {
        case COL:
            dx = 1;
            dy = 0;
            break;
        case DIA:
            dx = 1;
            dy = 1;
            break;
        case COU:
            dx = 1;
            dy = -1;
            break;
        default:
            break;
        }
        for (int i = 0; i < STONE_NUM; i++)
        {
            int row = i / BOARD_SIZE, col = i % BOARD_SIZE;
            UC act = ActionFlatten(row, col);
            if (ActionToAnchor(act, dir) != act) continue;
            U32** handle = new U32*[ACTION_TABLE_SIZE];
            rotatedActionCodingTable[anchorIndexTable[d][act]] = handle;
            for (int c = 0; c < ACTION_TABLE_SIZE; c++)
            {
                UC* actions = actionTable[c];
                U32 codings[BOARD_SIZE + 1] = { 0 };
                for (int i = 1; i <= actions[0]; i++)
                    codings[++codings[0]] = CodeAction(row + dx * static_cast<int>(actions[i]), 
                                                       col + dy * static_cast<int>(actions[i]));
                handle[c] = new U32[codings[0] + 1];
                memcpy(handle[c], codings, sizeof(U32) * (codings[0] + 1));
            }
        }
    }
}