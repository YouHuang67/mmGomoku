#include "pns.h"

VCTBoard* PNSNode::boardPtr = nullptr;
StoneType PNSNode::attacker;
PNSNode* PNSNode::seenNodes[STONE_NUM];
U64HashTable<PNSNode>* PNSNode::nodeTablePtr = nullptr;
U32 PNSNode::seenActPlys[STONE_NUM];
int PNSNode::forwardLevelNum;
int PNSNode::backwardLevelNum;
int PNSNode::leafCount;
int PNSNode::moveCount;
