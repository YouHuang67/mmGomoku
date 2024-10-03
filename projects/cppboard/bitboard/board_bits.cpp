#include "board_bits.h"

U32* InitializeMasks();
U32** InitializeBits();
U32** InitializeStones();

U32* Bit::masks = InitializeMasks();
U32** Bit::initialBits = InitializeBits();
U32** Bit::stones = InitializeStones();

void Bit::Initialize()
{
    for (int i = 0; i < 4; i++)
    {
        memcpy(bits[i], initialBits[i], sizeof(U32) * BOARD_SIZE);
    }
    step = 0;
}

Bit& Bit::operator=(const Bit& bit)
{
    for (int i = 0; i < 4; i++)
    {
        memcpy(bits[i], bit.bits[i], sizeof(U32) * BOARD_SIZE);
    }
    step = bit.step;
	memcpy(history, bit.history, sizeof(UC) * step);
    return *this;
}

Bit::Bit(const UC* his, int hisNum) 
{
    for (int i = 0; i < hisNum; i++)
    {
        Move(his[i]);
    }
}

U32* InitializeMasks()
{
	U32 *maskPointer = new U32[BOARD_SIZE];
	maskPointer[0] = MASK;
	for (int i = 1; i < BOARD_SIZE; i++)
	{
		maskPointer[i] = (maskPointer[i - 1] << 2) ^ MASK;
	}
	return maskPointer;
}

U32** InitializeBits()
{
	U32 **bitPointer = new U32*[4];
	U32 line = 0;
	for (int i = 0; i < BOARD_SIZE; i++)
	{
		line ^= ((~static_cast<U32>(EMPTY)) & MASK) << (2 * i);
	}
	for (int i = 0; i < 4; i++)
	{
		bitPointer[i] = new U32[BOARD_SIZE];
		for (int j = 0; j < BOARD_SIZE; j++)
		{
			bitPointer[i][j] = line;
		}
	}
	return bitPointer;
}

U32** InitializeStones()
{
	U32 **stonePointer = new U32*[2];
	U32 black = static_cast<U32>(BLACK);
    U32 white = static_cast<U32>(WHITE);
    U32 empty = static_cast<U32>(EMPTY);
	for (auto &player : { black, white })
	{
		stonePointer[static_cast<StoneType>(player)] = new U32[BOARD_SIZE];
		for (int i = 0; i < BOARD_SIZE; i++)
		{
			stonePointer[static_cast<StoneType>(player)][i] = 
                (((~player) ^ (~empty)) & MASK) << (2 * i);
		}
	}
	return stonePointer;
}

std::ostream& operator<<(std::ostream& os, const Bit& bit)
{
	os << "  ";
	for (int i = 0; i < BOARD_SIZE; i++)
	{
		os << " ";
		if (i < 10)
		{
			os << " ";
		}
		else
		{
			os << 1;
		}
	}
	os << endl << "  ";
	for (int i = 0; i < BOARD_SIZE; i++)
	{
		os << " " << i % 10;
	}
	os << endl;
	for (int i = 0; i < BOARD_SIZE; i++)
	{
		U32 line = bit.GetLine(ActionFlatten(i, 0), ROW);
		os << (i < 10 ? " " : "1") << i % 10;
		for (int j = 0; j < BOARD_SIZE; j++)
		{
			os << " ";
			StoneType ply = static_cast<StoneType>((~(line >> (2 * j))) & MASK);
			switch (ply)
			{
			case EMPTY:
				os << "_";
				break;
			case BLACK:
				os << "X";
				break;
			case WHITE:
				os << "O";
				break;
            default:
                cout << "Illegal bits " << ply << endl;
                exit(0);
			}
		}
		os << endl;
	}
	return os;
}

std::ostream& operator<<(std::ostream& os, const LinePrinter& lp)
{
	U32 line = lp.line;
    for (int i = 0; i < BOARD_SIZE; i++)
    {
        StoneType st = static_cast<StoneType>((~line) & MASK);
        switch (st)
        {
        case BLACK:
            os << "X";
            break;
        case WHITE:
            os << "O";
            break;
        case EMPTY:
            os << "_";
            break;
        default:
            break;
        }
        line >>= 2;
    }
    os << endl;
	return os;
}


