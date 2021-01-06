class Group:
    def __init__(self, idGroup, time, captured):
        self.idGroup = idGroup
        self.timestamp = time
        self.captured = captured

    def __repr__(self):
        return str(self.idGroup) + " time: " + str(self.timestamp) + "captured: " + str(self.captured)

def updateGroupList(oldList, newList):
    result = []
    
    for group in newList:
        overlapped = list(filter(lambda x: x.idGroup == group.idGroup, oldList))
        if not overlapped:
            result.append(group)
        else:
            assert(len(overlapped) == 1)
            result.append(overlapped[0])

    return result
