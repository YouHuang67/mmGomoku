#ifndef UTILS
#define UTILS

#include <cstring>
#include <iostream>
#include <cstdlib>

using std::memcpy;
using std::cout;
using std::endl;
using std::rand;

typedef unsigned long long U64;
typedef unsigned long int U32;
typedef unsigned char UC;
//bits
enum StoneType { BLACK = 0, WHITE, EMPTY, NONE };
enum Direction { ROW, COL, DIA, COU};
const unsigned int BOARD_SIZE = 15;
const unsigned int STONE_NUM = 225;
const U32 MASK = static_cast<U32>(3);
//shapes
enum ShapeType { FIVE = 0, OPEN_FOUR, FOUR, OPEN_THREE, THREE, OPEN_TWO, TWO };
const unsigned int CODING_LENGTH = 16;
const U32 CODING_MASK = (1 << CODING_LENGTH) - 1;
const unsigned int ACTION_TABLE_SIZE = 1 << BOARD_SIZE;
const unsigned int MAP_LENGTH = 10;
const unsigned int MAP_SIZE = 1 << (2 * MAP_LENGTH);
const U32 LINE_MASK = MAP_SIZE - 1;

struct TargetShape
{
    ShapeType target;
    unsigned int min;
    unsigned int max;
};
const TargetShape OPEN_FOUR_TARGET  = { FIVE, 2, BOARD_SIZE };
const TargetShape FOUR_TARGET       = { FIVE, 1, 1 };
const TargetShape OPEN_THREE_TARGET = { OPEN_FOUR, 1, BOARD_SIZE };
const TargetShape THREE_TARGET      = { FOUR, 1, BOARD_SIZE };
const TargetShape OPEN_TWO_TARGET   = { OPEN_THREE, 1, BOARD_SIZE };

//board
const unsigned int ACTION_CODING_LENGTH = 4;
const UC ACTION_CODING_MASK = (1 << ACTION_CODING_LENGTH) - 1;

inline
UC ActionFlatten(int row, int col)
{
    return static_cast<UC>((row << 4) ^ col);
} 

inline 
void ActionUnflatten(UC act, int& row, int& col)
{
    static UC mask = (1 << 4) - 1;
    row = static_cast<int>(act >> 4);
    col = static_cast<int>(mask & act);
}

inline
int Pow(int x, int n) 
{
    if (!n)
    {
        return 1;
    }
    return x * Pow(x, n-1);
}

inline
U64 Rand64()
{
    return static_cast<U64>(rand()) ^ 
        (static_cast<U64>(rand()) << 15) ^ 
        (static_cast<U64>(rand()) << 30) ^ 
        (static_cast<U64>(rand()) << 45) ^ 
        (static_cast<U64>(rand()) << 60);
}

inline
float FloatRand()
{
    static float max = static_cast<float>(RAND_MAX);
    return static_cast<float>(rand()) / max;
}

inline
U64* InitializeZobristKeys(int n)
{
    U64* ptr = new U64[n];
    for (int i = 0; i < n; i++) ptr[i] = Rand64();
    return ptr;
}

template<class HashType>
class U64HashTable
{
    public:
        U64HashTable(std::size_t s) : size(s) { Initialize(); }
        ~U64HashTable();
        HashType* Find(U64 key) const;
        HashType** FindHandle(U64 key);
        std::size_t Size() const { return pointer; }
        void DeleteAllObjects();
        void Reset();

    private:
        std::size_t size, pointer;
        U64* keyPtr;
        HashType** valuePtr;
        U64 hash;
        int* headPtr;
        int* nextPtr;
        void Initialize();
        void Rehash();

};

template<class HashType>
void U64HashTable<HashType>::Initialize()
{
    keyPtr = new U64[size]();
    valuePtr = new HashType*[size]();
    hash = 1;
    while ((hash << 1) < static_cast<U64>(size)) hash <<= 1;
    headPtr = new int[static_cast<std::size_t>(hash--)]();
    nextPtr = new int[size]();
    pointer = 0;
}

template<class HashType>
U64HashTable<HashType>::~U64HashTable()
{
    delete[] keyPtr;
    delete[] valuePtr;
    delete[] headPtr;
    delete[] nextPtr;
}

template<class HashType>
HashType* U64HashTable<HashType>::Find(U64 key) const
{
    int head = static_cast<int>(key & hash);
    for (int ptr = headPtr[head]; ptr != 0; ptr = nextPtr[ptr])
        if (keyPtr[ptr] == key) return valuePtr[ptr];
    return nullptr;
}

template<class HashType>
HashType** U64HashTable<HashType>::FindHandle(U64 key)
{
    int head = static_cast<int>(key & hash);
    for (int ptr = headPtr[head]; ptr != 0; ptr = nextPtr[ptr])
        if (keyPtr[ptr] == key) return valuePtr + ptr;
    if (pointer + 1 >= size)
    {   
        Rehash();
        head = static_cast<int>(key & hash);
    } 
    keyPtr[++pointer] = key;
    nextPtr[pointer] = headPtr[head];
    headPtr[head] = pointer;
    return valuePtr + pointer;
}

template<class HashType>
void U64HashTable<HashType>::DeleteAllObjects()
{
    for (int i = 1; i <= pointer; i++)
        delete valuePtr[i];
}

template<class HashType>
void U64HashTable<HashType>::Reset()
{
    memset(headPtr, 0, sizeof(int) * (hash + 1));
    memset(valuePtr, 0, sizeof(HashType*) * (pointer + 1));
    pointer = 0;
}

template<class HashType>
void U64HashTable<HashType>::Rehash()
{
    std::size_t preSize = size;
    size <<= 1;
    U64* preKeyPtr = keyPtr;
    HashType** preValuePtr = valuePtr;
    delete[] headPtr;
    delete[] nextPtr;
    Initialize();
    for (int i = 0; i < preSize; i++)
        if (preValuePtr[i]) (*FindHandle(preKeyPtr[i])) = preValuePtr[i];
    delete[] preKeyPtr;
    delete[] preValuePtr;
}

#endif