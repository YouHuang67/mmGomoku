#ifndef BOARD_BITS
#define BOARD_BITS


#include "utils.h"

class Bit
{
    public:
        void Initialize();
        Bit() { Initialize(); }
        Bit& operator=(const Bit& bit);
        Bit(const Bit& bit) { (*this) = bit; }
        Bit(const UC* his, int hisNum);
        void MetaMove(UC act, StoneType ply);
        void Move(UC act, StoneType ply = EMPTY);
        UC Undo();
        StoneType GetPlayer() const;
        StoneType GetPreviousPlayer() const;
        inline int GetStep() const { return step; }
        U32 GetLine(UC pos, Direction dir) const;
		inline static U32 GetMask(int length) { return masks[length]; }
		inline static U32 GetStone(StoneType ply, UC act) 
            { return stones[static_cast<int>(ply)][static_cast<int>(act)]; }
        inline bool IsLegal(UC act) const 
            { return static_cast<StoneType>(
                (~(bits[0][act >> 4] >> (2 * (act & 15)))) & 3) == EMPTY; }
        inline UC GetPreviousAction() const { return history[step - 1]; }
    
    private:
        static U32 *masks;
        static U32 **initialBits;
        static U32 **stones;
        U32 bits[4][BOARD_SIZE];
        StoneType players[STONE_NUM] = { BLACK };
        UC history[STONE_NUM] = {0};
        int step = 0;
        
};

class LinePrinter
{
    public:
        LinePrinter(U32 l): line(l) {}
    
    private:
        U32 line;
        friend std::ostream& operator<<(std::ostream& os, const LinePrinter& lp);
};

std::ostream& operator<<(std::ostream& os, const Bit& bit);

std::ostream& operator<<(std::ostream& os, const LinePrinter& lp);

inline
void Bit::MetaMove(UC act, StoneType ply) 
{
    int row, col;
    ActionUnflatten(act, row, col);
    int player = static_cast<int>(ply), gap;
	bits[0][row] ^= stones[player][col];
	bits[1][col] ^= stones[player][row];
	gap = row - col;
	bits[2][(gap + BOARD_SIZE) % BOARD_SIZE] ^= stones[player][col];
	gap = row - (BOARD_SIZE - 1 - col);
	bits[3][(gap + BOARD_SIZE) % BOARD_SIZE] ^= stones[player][BOARD_SIZE - 1 - col];
}

inline
void Bit::Move(UC act, StoneType ply) 
{
    if (ply == EMPTY)
    {
        ply = static_cast<StoneType>(step % 2);
    }
    MetaMove(act, ply);
    players[step] = ply;
    history[step++] = act;
}

inline
UC Bit::Undo() 
{
    --step;
    MetaMove(history[step], players[step]);
    return history[step];
}

inline
U32 Bit::GetLine(UC pos, Direction dir) const 
{
    int row, col;
    ActionUnflatten(pos, row, col);
	int gap;
	switch (dir)
	{
	case ROW:
		return bits[0][row];
	case COL:
		return bits[1][col];
	case DIA:
		gap = row - col;
		break;
	case COU:
		gap = row - (BOARD_SIZE - 1 - col);
		break;
    default:
        cout << "Illegal Direction: " << dir << endl;
        exit(0);
	}
	if (gap >= 0)
	{
		return bits[static_cast<int>(dir)][gap] & masks[BOARD_SIZE - 1 - gap];
	}
	else
	{
		return bits[static_cast<int>(dir)][gap + BOARD_SIZE] >> (-2 * gap);
	}
}

inline
StoneType Bit::GetPlayer() const 
{ 
    if (!step) return BLACK;
    if (BLACK == players[step - 1]) return WHITE;
    else return BLACK;
}

inline
StoneType Bit::GetPreviousPlayer() const 
{ 
    if (!step) return BLACK;
    else return players[step - 1];
}

#endif