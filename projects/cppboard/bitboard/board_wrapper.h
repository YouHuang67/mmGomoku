#ifndef BOARD_WRAPPER
#define BOARD_WRAPPER

#include <vector>
#include "vct_utils.h"
#include "pns.h"

typedef std::vector<int> IntVector;

class BoardWrapper
{
    public:
        BoardWrapper() {}
        BoardWrapper(const BoardWrapper& boardWrapper) 
            { memcpy(this, &boardWrapper, sizeof(BoardWrapper)); }
        IntVector Evaluate(int maxNodeNum = 100000);
        IntVector GetActions(int ply, int sp, int it) const;
        void Move(int act);
        bool IsOver() const { return board.IsOver(); }
        int Attacker() const { return static_cast<int>(attacker); }
        int Winner() const { return static_cast<int>(board.Winner()); }
        int Player() const { return static_cast<int>(board.GetPlayer()); }
        IntVector Key() const;
        IntVector NextKey(int act) const;
        IntVector BoardVector() const;
        static IntVector HomogenousActions(int act);

    private:
        VCTBoard board;
        StoneType attacker = EMPTY;
        PNSNode* vctNode = nullptr;
        static IntVector UCsToInts(UC* UCs);
        static void RotateAction90(int& row, int& col);

};

inline
void BoardWrapper::Move(int act)
{
    UC action = static_cast<UC>(act);
    board.Move(action);
    if (vctNode)
    {
        vctNode = vctNode->Next(action);
        if (!vctNode) attacker = EMPTY;
    } 
}

inline
IntVector BoardWrapper::Evaluate(int maxNodeNum)
{
    StoneType player = board.GetPlayer();
    static UC actions[STONE_NUM + 1];
    actions[0] = 0;
    if (EMPTY == attacker)
    {
        vctNode = PNSVCT(board, player, maxNodeNum);
        if (vctNode) attacker = player;
    }
    IntVector vec;
    if (player == attacker && vctNode)
    {
        UC action = vctNode->GetAttackAction();
        if (action != 0xff)
        {   
            vec.push_back(static_cast<int>(action));
            return vec;
        }
    }
    if (board.GetActions(player, OPEN_FOUR, true, actions))
        return UCsToInts(actions);
    if (board.GetActions(player, FOUR, true, actions))
        return UCsToInts(actions);
    if (board.GetActions(player, OPEN_FOUR, false, actions))
        return UCsToInts(actions);
    if (board.GetActions(player, FOUR, false, actions))
        return UCsToInts(actions);
    if (board.GetActions(player, OPEN_THREE, true, actions))
        return UCsToInts(actions);
    if (board.GetActions(player, OPEN_THREE, false, actions)) 
    {
        board.GetActions(player, THREE, true, actions);
        FilterReplicatedActions(actions);
        return UCsToInts(actions);
    }
    if (POSITIVE == board.Evaluate(player, actions)) 
        return UCsToInts(actions);
    else return vec;
}

inline
IntVector BoardWrapper::GetActions(int ply, int sp, int it) const
{
    IntVector vec;
    static UC actions[STONE_NUM + 1];
    actions[0] = 0;
    if (board.GetActions(static_cast<StoneType>(ply), 
                         static_cast<ShapeType>(sp), 
                         static_cast<bool>(it), actions))
        vec = UCsToInts(actions);
    return vec;
}

inline
IntVector BoardWrapper::UCsToInts(UC* UCs)
{
    IntVector vec(UCs[0], 0);
    for (int i = 1; i <= UCs[0]; i++) 
        vec[i - 1] = static_cast<int>(UCs[i]);
    return vec;
}

inline
IntVector BoardWrapper::Key() const
{
    U64 key = board.Key();
    static const std::size_t blockNum = 4, blockSize = 16;
    static const U64 blockMask = (1 << blockSize) - 1;
    IntVector keyVector(4, 0);
    for (int i = 0; i < blockNum; i++)
        keyVector[i] = static_cast<int>((key >> (i * blockSize)) & blockMask);
    return keyVector;
}

inline
IntVector BoardWrapper::NextKey(int act) const
{
    U64 key = board.UpdateZobristKey(
        board.Key(), board.GetPlayer(), static_cast<UC>(act)
    );
    static const std::size_t blockNum = 4, blockSize = 16;
    static const U64 blockMask = (1 << blockSize) - 1;
    IntVector keyVector(4, 0);
    for (int i = 0; i < blockNum; i++)
        keyVector[i] = static_cast<int>((key >> (i * blockSize)) & blockMask);
    return keyVector;
}

inline
IntVector BoardWrapper::BoardVector() const
{
    IntVector vec(STONE_NUM, 0);
    int index = 0;
    for (int row = 0; row < BOARD_SIZE; row++)
    {
        U32 line = board.GetLine(static_cast<UC>(row) << 4, 0);
        for (int col = 0; col < BOARD_SIZE; col++)
            vec[index++] = (~(line >> (2 * col))) & MASK;
    }
    return vec;
}

inline
void BoardWrapper::RotateAction90(int& row, int& col)
{
    int temp = col;
    col = BOARD_SIZE - 1 - row;
    row = temp;
}

inline
IntVector BoardWrapper::HomogenousActions(int act)
{
    IntVector vec(8, 0);
    int index = 0;
    int row, col;
    ActionUnflatten(act, row, col);
    vec[index++] = act;
    RotateAction90(row, col);
    for (int i = 0; i < 3; i++)
    {
        vec[index++] = ActionFlatten(row, col);
        RotateAction90(row, col);
    }
    vec[index++] = ActionFlatten(BOARD_SIZE - 1 - row, col);
    vec[index++] = ActionFlatten(row, BOARD_SIZE - 1 - col);
    vec[index++] = ActionFlatten(col, row);
    RotateAction90(row, col);
    row = row + col;
    col = row - col;
    row = row - col;
    for (int i = 0; i < 3; i++) RotateAction90(row, col);
    vec[index++] = ActionFlatten(row, col);
    return vec;
}   

#endif