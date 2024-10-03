#ifndef SHAPE
#define SHAPE

#include "utils.h"

class LineShapeBase {};

class LineShape : public LineShapeBase
{
	public:
		LineShape() { Initialize(); }
		LineShape(const U32 &l) { Initialize(l); }
		~LineShape() { existedShapeCount--; }
		void reset() { Initialize(); }
		U32 GetActionCoding(ShapeType shape, bool isAttacker) const;
		bool GetActions(ShapeType shape, bool isAttacker, 
						UC* actions = nullptr) const;
		void SetActions(ShapeType shape, bool isAttacker, U32 actionCoding);
		U32 GetLine() const { return _line; }
		inline void SetLine(U32 l) { _line = l; }
		void SetFive(bool isFive) { _isFive = isFive; }
		inline bool IsFive() const { return _isFive; }
		inline static unsigned int GetShapeCount() { return shapeCount; }
		inline static unsigned int GetExistedShapeCount() { return existedShapeCount; }
		inline static UC** GetActionTable() { return actionTable; }

	private:
		U32 _line;
		bool _isFive;
		U32 _actions[5];
		void Initialize(U32 l = 0);
		static UC** actionTable;
		static unsigned int shapeCount;
		static unsigned int existedShapeCount;

};

class LineShapeMap : public LineShapeBase
{
	public:
		LineShapeMap() {}
		LineShapeMap(std::size_t mapSize);
		~LineShapeMap() { delete[] mapPtr; }
		inline LineShapeBase* Find(U32 key) { return *FindHandle(key); }
		inline LineShapeBase** FindHandle(U32 key) { return mapPtr + lineIndices[key]; }
		inline static int GetLineIndex(U32 line) { return lineIndices[line]; }
		inline static int GetLineLength(U32 line) { return lineLengths[line]; }
		inline static U32 GetLineMask(int length) { return lineMasks[length]; }
		inline const std::size_t Size() { return size; }
		
	private:
		static int* lineIndices;
		static int* lineLengths;
		static U32* lineMasks;
		std::size_t size;
		LineShapeBase** mapPtr = nullptr;

};

class LineShapeTable
{
	public: 
		static LineShapeTable* Get();
		inline static LineShape* Find(U32 key) { return static_cast<LineShape*>(*(FindHandle(key))); }
		static LineShapeBase** FindHandle(U32 key);
		inline static LineShape* End() { return endPtr; }

	private:
		LineShapeTable() {};
		LineShapeTable(const LineShapeTable&) = delete;
		LineShapeTable& operator=(const LineShapeTable&) = delete;
		static LineShapeTable* lineshapeTablePointer;
		static LineShapeBase* tablePtr;
		static LineShape* endPtr;

};

inline
U32 CodeToLine(int code, int length)
{
	U32 line = 0;
	for (int i = 0; i < length; i++)
	{
		U32 st = code % 3;
		line = (line << 2) ^ (MASK & (~st));
		code /= 3;
	}
	return line;
}

inline
U32 LineShape::GetActionCoding(ShapeType shape, bool isAttacker) const
{
	static U32 actionMask = (1 << CODING_LENGTH) - 1;
	if (isAttacker) return _actions[static_cast<int>(shape)-1] >> CODING_LENGTH; 
	else            return _actions[static_cast<int>(shape)-1] & actionMask;
}

inline
bool LineShape::GetActions(ShapeType shape, bool isAttacker, UC* actions) const
{
	U32 actionCoding = GetActionCoding(shape, isAttacker);
	if (actions) 
	{
		UC* handle = actionTable[actionCoding];
		memcpy(actions, handle, sizeof(UC) * (handle[0] + 1));
	}
	return actionCoding != 0;
}

inline
void LineShape::SetActions(ShapeType shape, bool isAttacker, U32 actionCoding)
{
	if (isAttacker) actionCoding <<= CODING_LENGTH;
	_actions[static_cast<int>(shape)-1] ^= actionCoding;
}

inline
LineShapeBase** LineShapeTable::FindHandle(U32 key)
{
	LineShapeBase** ptr = &tablePtr;
	while (true)
	{
		U32 line = LINE_MASK & key;
		ptr = static_cast<LineShapeMap*>(*ptr)->FindHandle(line);
		if (LineShapeMap::GetLineLength(line) < MAP_LENGTH) return ptr;
		key >>= 2 * MAP_LENGTH;
	}
}

#endif