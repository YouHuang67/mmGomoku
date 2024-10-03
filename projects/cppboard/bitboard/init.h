#ifndef INIT
#define INIT

#include <unordered_map>
#include "utils.h"
#include "lineshapes.h"
#include "shapes.h"

class LineGenerator
{
    public:
        LineGenerator(): index(0), length(0), totalNum(0) {}
        int SetLength(int l);
        U32 step();

    private:
        int index;
        int length;
        int totalNum;

};

class ActionHash
{
    public:
        static ActionHash* Get();
        static U64 Hash(const LineShape* lineShape);

    private:
        ActionHash();
		ActionHash(const ActionHash&) = delete;
		ActionHash& operator=(const ActionHash&) = delete;
        static ActionHash* actionHashingPointer;
        static U64* lengthZobristKey;
        static U64* fiveZobristKey;
        static U64* zobristKeys[5][2];

};

class LineShapeInitialization
{
    public:
        LineShapeInitialization(int verbose = 0);

    private:
        Shape shape;
        LineShapeTable* tablePtr;
        LineGenerator lineGenerator;
        void InitializeShapeLengthByLength();
        void ResetShape();
        bool Load(int verbose);
        bool Save(int verbose);
        static bool isLoaded;

};

inline
int LineGenerator::SetLength(int l) 
{ 
    index = 0;
    length = l;
    totalNum = Pow(3, l); 
    return totalNum;
}

inline
U32 LineGenerator::step()
{
    U32 line = 0;
    if (index >= totalNum)
    {
        index = 0;
        totalNum = 0;
        return line;
    }
    int code = index;
    for (int i = 0; i < length; i++)
    {
        line <<= 2;
        line ^= MASK & (~static_cast<U32>(code % 3));
        code /= 3;
    }
    index++;
    return line;
}

inline
int GetLineLength(U32 line)
{
    int length = LineShapeMap::GetLineLength(LINE_MASK & line);
    line >>= MAP_LENGTH;
    while (line)
    {
        length += LineShapeMap::GetLineLength(LINE_MASK & line);
        line >>= MAP_LENGTH;
    }
    return length;
}

inline 
U64 ActionHash::Hash(const LineShape* lineShape) 
{
    static int shapeStart = static_cast<int>(OPEN_FOUR);
    static int shapeEnd = static_cast<int>(OPEN_TWO);
    static int isAttackerStart = static_cast<int>(false);
    static int isAttackerEnd = static_cast<int>(true);
    U64 key = lengthZobristKey[GetLineLength(lineShape->GetLine())];
    key ^= fiveZobristKey[static_cast<int>(lineShape->IsFive())];
    for (int sp = shapeStart; sp <= shapeEnd; sp++)
    {
        int i = sp - shapeStart;
        ShapeType shape = static_cast<ShapeType>(sp);
        for (int j = isAttackerStart; j <= isAttackerEnd; j++)
        {
            bool isAttacker = static_cast<bool>(j);
            UC actions[BOARD_SIZE + 1];
            if (lineShape->GetActions(shape, isAttacker, actions))
                for (int k = 1; k <= actions[0]; k++) key ^= zobristKeys[i][j][actions[k]];
        }
    }
    return key;
}

#endif