#include <iostream>
#include <fstream>
#include "init.h"

ActionHash* ActionHash::actionHashingPointer = nullptr;
U64* ActionHash::lengthZobristKey;
U64* ActionHash::fiveZobristKey;
U64* ActionHash::zobristKeys[5][2];

bool LineShapeInitialization::isLoaded = false;

ActionHash* ActionHash::Get() 
{
    if (!actionHashingPointer) actionHashingPointer = new ActionHash;
    return actionHashingPointer;
}

ActionHash::ActionHash() 
{
    std::srand(0);
    lengthZobristKey = InitializeZobristKeys(BOARD_SIZE + 1);
    fiveZobristKey = InitializeZobristKeys(2);
    int shapeStart = static_cast<int>(OPEN_FOUR);
    int shapeEnd = static_cast<int>(OPEN_TWO);
    int isAttackerStart = static_cast<int>(false);
    int isAttackerEnd = static_cast<int>(true);
    for (int sp = shapeStart; sp <= shapeEnd; sp++)
    {
        int i = sp - shapeStart;
        for (int j = isAttackerStart; j <= isAttackerEnd; j++)
            zobristKeys[i][j] = InitializeZobristKeys(BOARD_SIZE);
    }
}

LineShapeInitialization::LineShapeInitialization(int verbose) 
{
    if (isLoaded) return;
    tablePtr = LineShapeTable::Get();
    if (Load(verbose)) 
    {
        isLoaded = true;
        if (verbose) cout << "Load shapes successfully" << endl;
        return;
    }
    if (verbose) cout << "Failed to load shapes, generating..." << endl;
    InitializeShapeLengthByLength();
    ResetShape();
    if (Save(verbose)) isLoaded = true; 
    else cout << "Failed to save shapes" << endl;
}

void LineShapeInitialization::InitializeShapeLengthByLength()
{
    for (int length = 1; length <= BOARD_SIZE; length++)
    {
        lineGenerator.SetLength(length);
        U32 line = lineGenerator.step();
        while (line)
        {
            shape.FindAllShapes(line);
            line = lineGenerator.step();
        }
    }
}

void LineShapeInitialization::ResetShape()
{
    ActionHash* actionHashPtr = ActionHash::Get();
    U64HashTable<LineShape> lineShapeSet(1 << 20);
    for (int length = 1; length <= BOARD_SIZE; length++)
    {
        lineGenerator.SetLength(length);
        U32 line = lineGenerator.step();
        while (line)
        {
            LineShapeBase** baseHandle = tablePtr->FindHandle(line);
            if (!(*baseHandle))
            {
                line = lineGenerator.step();
                continue;
            }
            LineShape* shapePtr = static_cast<LineShape*>(*baseHandle);
            U64 key = actionHashPtr->Hash(shapePtr);
            LineShape** setHandle = lineShapeSet.FindHandle(key);
            if (*setHandle) *baseHandle = *setHandle;
            else *setHandle = shapePtr;
            line = lineGenerator.step();
        }
    }
}

bool LineShapeInitialization::Save(int verbose)
{
    std::ofstream outFile("lineshapes.dat", std::ios::out | std::ios::binary);
    if (!outFile) return false;
    for (int length = 1; length <= BOARD_SIZE; length++)
    {
        lineGenerator.SetLength(length);
        U32 line = lineGenerator.step();
        while (line)
        {
            const LineShape* handle = shape.FindAllShapes(line);
            if (!handle)
            {
                line = lineGenerator.step();
                continue;
            }
            U32 targetLine = handle->GetLine();
            outFile.write((char*)&line, sizeof(U32));
            outFile.write((char*)&targetLine, sizeof(U32));
            if (line == targetLine) outFile.write((char*)handle, sizeof(LineShape));
            line = lineGenerator.step();
        }
    }
    return true;
}

bool LineShapeInitialization::Load(int verbose)
{
    std::ifstream inFile("lineshapes.dat", std::ios::in | std::ios::binary);
    if (!inFile) return false;
    U32 line, targetLine;
    while (inFile.read((char*)&line, sizeof(U32)))
    {
        inFile.read((char*)&targetLine, sizeof(U32));
        LineShapeBase** handle = tablePtr->FindHandle(line);
        LineShapeBase** targetHandle = tablePtr->FindHandle(targetLine);
        if (!(*targetHandle)) *targetHandle = new LineShape;
        if (line == targetLine) inFile.read((char*)*targetHandle, sizeof(LineShape));
        else *handle = *targetHandle;
    }
    inFile.close();

    LineShape* nullLineShape = LineShapeTable::End();
    for (int length = 1; length <= BOARD_SIZE; length++)
    {
        lineGenerator.SetLength(length);
        U32 line = lineGenerator.step();
        while (line)
        {
            LineShapeBase** handle = tablePtr->FindHandle(line);
            if (!(*handle)) *handle = nullLineShape;
            line = lineGenerator.step();
        }
        if (verbose) 
            cout << length << "-length line shapes have been loaded" << endl;
    }
    return true;
}
