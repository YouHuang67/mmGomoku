#ifndef NODE
#define NODE

#include "utils.h"
#include "memory.h"

const unsigned int ARRAY_INIT_SIZE = 1 << 20;
const unsigned int PARENTS_INIT_SIZE = 4;

template<typename NodeType>
class VCTNode
{
    public:
        VCTNode() {};
        void Reset();
        UC Action(int index) const { return Actions()[index + 1]; }
        UC* Actions() const; // the first element is the number of actions
        unsigned int ActionsNum() const;
        unsigned int NodeIndex() const { return nodeIndex; }
        unsigned int ChildrenNum() const { return childrenNum; }
        unsigned int ParentsNum() const { return parentsNum; }
        NodeType* Child(int index) const;
        NodeType* Parent(int index) const;
        void SetNodeIndex(int index) { nodeIndex = index; }
        void SetActions(UC* acts);
        void SetChild(NodeType* nodePtr, int index = -1);
        void SetParent(NodeType* nodePtr, int index = -1);
        static NodeType* Node(int index) { return nodeArrayPtr->GetItem(index); }
        static NodeType* GetNode();
        static void InitializeArrays();
        static void ResetArrays();
        static unsigned int GetNodeCount() { return nodeCount; }

    private:
        static bool initialized;
        static DynamicArray<NodeType>* nodeArrayPtr;
        static DynamicArray<int>* nodePointerArrayPtr;
        static DynamicArray<UC>* actionArrayPtr;
        static unsigned int nodeCount;
        int nodeIndex;
        int actionIndice;
        int childrenIndice;
        int parentsIndice;
        int childrenNum;
        int parentsSize;
        int parentsNum;
        
};

template<typename NodeType>
void VCTNode<NodeType>::InitializeArrays()
{
    if (!initialized)
    {
        nodeArrayPtr = new DynamicArray<NodeType>(ARRAY_INIT_SIZE);
        nodePointerArrayPtr = new DynamicArray<int>(ARRAY_INIT_SIZE);
        actionArrayPtr = new DynamicArray<UC>(ARRAY_INIT_SIZE);
        initialized = true;
    }
}

template<typename NodeType>
void VCTNode<NodeType>::Reset()
{
    actionIndice = -1;
    childrenIndice = -1;
    parentsIndice = -1;
    childrenNum = 0;
    parentsSize = 0;
    parentsNum = 0;
}

template<typename NodeType>
void VCTNode<NodeType>::SetActions(UC* acts)
{
    int actionsNum = acts[0];
    if (!actionsNum) return;
    actionIndice = actionArrayPtr->AllocateNewArray(actionsNum + 1);
    memcpy(Actions(), acts, sizeof(UC) * (actionsNum + 1));
}

template<typename NodeType>
unsigned int VCTNode<NodeType>::ActionsNum() const 
{
    if (actionIndice < 0) return 0;
    return actionArrayPtr->GetItem(actionIndice)[0];
}

template<typename NodeType>
UC* VCTNode<NodeType>::Actions() const 
{
    if (actionIndice < 0) return nullptr;
    return actionArrayPtr->GetItem(actionIndice);
}

template<typename NodeType>
NodeType* VCTNode<NodeType>::Child(int index) const 
{
    int nidx = nodePointerArrayPtr->GetItem(childrenIndice)[index];
    return Node(nidx);
}

template<typename NodeType>
void VCTNode<NodeType>::SetChild(NodeType* nodePtr, int index) 
{
    if (index < 0) index = childrenNum++;
    if (!index) 
        childrenIndice = nodePointerArrayPtr->AllocateNewArray(ActionsNum());
    nodePointerArrayPtr->GetItem(childrenIndice)[index] = nodePtr->NodeIndex();
}

template<typename NodeType>
NodeType* VCTNode<NodeType>::Parent(int index) const 
{
    int nidx = nodePointerArrayPtr->GetItem(parentsIndice)[index];
    return Node(nidx);
}

template<typename NodeType>
void VCTNode<NodeType>::SetParent(NodeType* nodePtr, int index) 
{
    if (index < 0) index = parentsNum++;
    if (index >= parentsSize)
    {
        if (parentsSize) 
        {
            while (index >= parentsSize) parentsSize <<= 1;
            int* handle = nodePointerArrayPtr->GetItem(parentsIndice);
            parentsIndice = nodePointerArrayPtr->AllocateNewArray(parentsSize);
            memcpy(nodePointerArrayPtr->GetItem(parentsIndice), 
                   handle, sizeof(int) * parentsNum);
        }
        else 
        {
            parentsSize = PARENTS_INIT_SIZE;
            parentsIndice = nodePointerArrayPtr->AllocateNewArray(parentsSize);
        }
    }
    nodePointerArrayPtr->GetItem(parentsIndice)[index] = nodePtr->NodeIndex();
}

template<typename NodeType>
NodeType* VCTNode<NodeType>::GetNode()
{
    int nodeIndex = nodeArrayPtr->AllocateNewObject();
    NodeType* nodePtr = Node(nodeIndex);
    nodePtr->Reset();
    nodePtr->SetNodeIndex(nodeIndex);
    nodeCount++;
    return nodePtr;
}

template<typename NodeType>
void VCTNode<NodeType>::ResetArrays()
{
    nodeArrayPtr->Reset();
    nodePointerArrayPtr->Reset();
    actionArrayPtr->Reset();
}

template<typename NodeType>
bool VCTNode<NodeType>::initialized = false;

template<typename NodeType>
DynamicArray<NodeType>* VCTNode<NodeType>::nodeArrayPtr = nullptr;

template<typename NodeType>
DynamicArray<int>* VCTNode<NodeType>::nodePointerArrayPtr = nullptr;
    
template<typename NodeType>
DynamicArray<UC>* VCTNode<NodeType>::actionArrayPtr = nullptr;

template<typename NodeType>
unsigned int VCTNode<NodeType>::nodeCount = 0;

#endif