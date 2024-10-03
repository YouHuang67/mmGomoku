#include "lineshapes.h"

UC** InitializeActionTable();

UC** LineShape::actionTable = InitializeActionTable();
unsigned int LineShape::shapeCount = 0;
unsigned int LineShape::existedShapeCount = 0;

int* GetLineIndices();
int* GetLineLengths();
U32* GetLineMasks();

int* LineShapeMap::lineIndices = GetLineIndices();
int* LineShapeMap::lineLengths = GetLineLengths();
U32* LineShapeMap::lineMasks = GetLineMasks();

LineShapeBase* GetTable();

LineShapeTable* LineShapeTable::lineshapeTablePointer = nullptr;
LineShapeBase* LineShapeTable::tablePtr = GetTable();
LineShape* LineShapeTable::endPtr = new LineShape;

UC** InitializeActionTable()
{
    UC** tablePtr = new UC*[ACTION_TABLE_SIZE];
    UC actions[BOARD_SIZE + 1];
    for (int i = 0; i < ACTION_TABLE_SIZE; i++)
    {
        U32 coding = i;
        int pos = 0;
        actions[0] = 0;
        while (coding)
        {
            if ((coding & 1) != 0) actions[++actions[0]] = pos;
            pos++;
            coding >>= 1;
        }
        tablePtr[i] = new UC[static_cast<std::size_t>(actions[0])+1];
        memcpy(tablePtr[i], actions, sizeof(UC) * (actions[0] + 1));
    }
    return tablePtr;
}

void LineShape::Initialize(U32 l) 
{
    _line = l;
    _isFive = false;
    memset(_actions, 0, sizeof(U32) * 5);
    shapeCount++;
    existedShapeCount++;
}

LineShapeMap::LineShapeMap(std::size_t mapSize)
{
    mapPtr = new LineShapeBase*[mapSize];
    for (int i = 0; i < mapSize; i++)
    {
        mapPtr[i] = nullptr;
    }
    size = mapSize;
}

inline int GetLengthOfLine(int l)
{
    U32 line = static_cast<U32>(l);
    int length = 0;
    while (line)
    {
        if (!(MASK & line))
        {
            return -1;
        }
        line >>= 2;
        length++;
    }
    return length;
}

int* GetLineIndices()
{
	int* indices = new int[MAP_SIZE];
	int index = 0;
	for (int i = 0; i < MAP_SIZE; i++)
	{
		if (GetLengthOfLine(i) >= 0)
		{
			indices[i] = index++;
		}
		else
		{
			indices[i] = -1;
		}
	}
	return indices;
}

int* GetLineLengths()
{
	int* lengths = new int[MAP_SIZE];
	for (int i = 0; i < MAP_SIZE; i++)
	{
		lengths[i] = GetLengthOfLine(i);
	}
	return lengths;
}

U32* GetLineMasks()
{
	U32* masks = new U32[BOARD_SIZE + 1];
	for (int i = 1; i <= BOARD_SIZE; i++)
	{
		masks[i] = (1 << (2 * i)) - 1;
	}
	return masks;
}

LineShapeTable* LineShapeTable::Get()
{
    if (lineshapeTablePointer) return lineshapeTablePointer;
    lineshapeTablePointer = new LineShapeTable;
    for (int length = 0; length < 5; length++)
    {
        int caseNum = Pow(3, length);
        for (int code = 0; code < caseNum; code++) 
            *FindHandle(CodeToLine(code, length)) = endPtr;
    }

    U32 line = 0;
    for (int i = 0; i < 5; i++) 
        line = (line << 2) ^ (MASK & (~static_cast<U32>(BLACK)));
    LineShape* ptr = new LineShape(line);
    ptr->SetFive(true);
    *FindHandle(line) = ptr;
    return lineshapeTablePointer;
}

void RecursivelySetMap(U32 line, LineShapeMap* map, int length = BOARD_SIZE)
{
    U32 key = LINE_MASK & line;
    if (LineShapeMap::GetLineLength(key) < MAP_LENGTH) return;
    LineShapeBase** handle = map->FindHandle(key);
    int leftLength = length - MAP_LENGTH;
    if (!(*handle))
    {
        int minLength = leftLength < MAP_LENGTH ? leftLength : MAP_LENGTH;
        U32 mask = (1 << (2 * minLength)) - 1;
        *handle = new LineShapeMap(LineShapeMap::GetLineIndex(mask)+1);
    }
    RecursivelySetMap(line >> (2 * MAP_LENGTH), 
                      static_cast<LineShapeMap*>(*handle), leftLength);
}

LineShapeBase* GetTable()
{
    LineShapeBase* ptr = static_cast<LineShapeBase*>(
        new LineShapeMap(LineShapeMap::GetLineIndex(LINE_MASK)+1));
    
    for (int length = 1; length <= BOARD_SIZE; length++)
    {
        int caseNum = Pow(3, length);
        for (int code = 0; code < caseNum; code++) 
        {
            RecursivelySetMap(CodeToLine(code, length), 
                              static_cast<LineShapeMap*>(ptr));
        }
    }

    return ptr;
}
 
