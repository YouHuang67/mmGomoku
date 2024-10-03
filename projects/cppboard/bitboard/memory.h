#ifndef MEMORY
#define MEMORY

#include <cstddef>

template<class Type>
class DynamicArray
{
    public:
        DynamicArray(std::size_t initSize) { array = new Type[(size = initSize)]; }
        ~DynamicArray() { delete[] array; }
        unsigned int AllocateNewArray(std::size_t s);
        unsigned int AllocateNewObject() { return AllocateNewArray(1); }
        void Reset() { ptr = 0; }
        inline Type* GetItem(unsigned int i) { return array + i; }

    private:
        unsigned int size, ptr = 0;
        Type* array;

};

template<class Type>
unsigned int DynamicArray<Type>::AllocateNewArray(std::size_t s)
{
    if (ptr + s > size)
    {
        unsigned int preSize = size;
        while (ptr + s > size) size <<= 1;
        Type* handle = array;
        array = new Type[size];
        memcpy(array, handle, sizeof(Type) * preSize);
        delete[] handle;
    }
    ptr += s;
    return ptr - s;
}

#endif