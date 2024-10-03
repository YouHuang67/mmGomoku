#ifndef SHAPES
#define SHAPES

#include "utils.h"
#include "board_bits.h"
#include "lineshapes.h"

class Shape
{
    public:
        bool Find(U32 line, StoneType player, ShapeType shape, 
                  bool isAttacker, UC* actions = nullptr);
        const LineShape* FindAllShapes(U32 line);
        static U32 WhiteToBlack(U32 whiteLine);

    private:
        LineShapeTable* shapeTablePtr = LineShapeTable::Get();
        LineShape* MetaFindAllShapes(U32 line);
        static U32* whiteToBlackTable;
        U32 MetaFind(U32 line, ShapeType targetShape, bool isAttacker);
        bool Check(U32 line, StoneType player, ShapeType targetShape, 
                   bool isAttacker);

};

inline 
bool Shape::Find(U32 line, StoneType player, ShapeType shape, 
                 bool isAttacker, UC* actions)
{
    if (actions) actions[0] = 0;
    if (WHITE == player) line = WhiteToBlack(line);
    const LineShape* mapPtr = FindAllShapes(line);
    if (!mapPtr) return false;
    if (FIVE == shape) return mapPtr->IsFive();
    return mapPtr->GetActions(shape, isAttacker, actions);
}

inline
const LineShape* Shape::FindAllShapes(U32 line)
{
    LineShapeBase** handle = shapeTablePtr->FindHandle(line);
    if (!(*handle)) *handle = MetaFindAllShapes(line);
    if (LineShapeTable::End() == (*handle)) return nullptr;
    return static_cast<LineShape*>(*handle);
}

inline
U32 Shape::WhiteToBlack(U32 whiteLine)
{
    U32 blackLine = 0;
    int length = 0;
    do
    {
        U32 line = LINE_MASK & whiteLine;
        blackLine ^= whiteToBlackTable[line] << (2 * length);
        length += LineShapeMap::GetLineLength(line);
        whiteLine >>= 2 * MAP_LENGTH;
    } while (whiteLine);
    return blackLine;
}

inline
bool Shape::Check(U32 line, StoneType player, ShapeType targetShape, 
                  bool isAttacker)
{
    if (Find(line >> 2, player, targetShape, isAttacker)) return true;
    int length = LineShapeMap::GetLineLength(LINE_MASK & line);
    U32 l = line >> (2 * MAP_LENGTH);
    while (l)
    {
        length += LineShapeMap::GetLineLength(LINE_MASK & l);
        l >>= 2 * MAP_LENGTH;
    }
    line &= LineShapeMap::GetLineMask(length - 1);
    return Find(line, player, targetShape, isAttacker);
}

#endif