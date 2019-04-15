from scapy.all import *
import numpy
import sys
import socket
import os
import csv

def packetExtraction(x, flowList, flowLabel):
    '''
    Extracts packet data to determine flows in the network, and adds each flow
    to the list of flows.
    '''

    # Output each packet to ensure sniff() is functional
    print(x.sprintf("{IP:%IP.src%,%IP.dst%,}"
                    "{TCP:%TCP.sport%,%TCP.dport%}"
                    "{UDP:%UDP.sport%,%UDP.dport%}"))

    # Assign packet values to variables
    srceIP = x.sprintf("{IP:%IP.src%}")
    srcePort = x.sprintf("{TCP:%TCP.sport%}{UDP:%UDP.sport%}")
    destIP = x.sprintf("{IP:%IP.dst%}")
    destPort = x.sprintf("{TCP:%TCP.dport%}{UDP:%UDP.dport%}")
    # Protocol is 0 for TCP, 1 for UDP
    protocol = 0
    if (x.sprintf("{TCP:tcp}{UDP:udp}") == "udp"): protocol = 1
    timeReceived = float(x.time)
    try:
        numBytes = int(x.sprintf("{IP:%IP.len%}"))
    except ValueError:
        numBytes = 0

    # If first flow, append to list
    if (len(flowList) == 0):
        # Create new flow with following format:
        # [source_IP (0), source_Port (1), destination_IP (2),
        # destination_Port (3), protocol (4), sent_Packets (5), sent_Bytes (6),
        # received_Packets (7), received_Bytes (8), total_Packets (9),
        # total_Bytes (10), time_Start (11), duration (12), flow_label (13)]
        newFlow = [srceIP, srcePort, destIP, destPort, protocol, 1, numBytes, 0,
                   0, 1, numBytes, timeReceived, timeReceived, flowLabel]
        flowList.append(newFlow)

    # Else, update numbers for the flow in the list
    else:
        # Variable to check if flow is in list
        isNewFlow = True

        for flow in flowList:
            # For sent packets, update sent & total numbers
            if all([srceIP == flow[0], srcePort == flow[1], destIP == flow[2],
                    destPort == flow[3], protocol == flow[4]]):
                flow[5] += 1
                flow[6] += numBytes
                flow[9] += 1
                flow[10] += numBytes
                flow[12] = timeReceived - flow[11]
                isNewFlow = False

            # For received packets, update received & total numbers
            elif all([srceIP == flow[2], srcePort == flow[3], destIP == flow[0],
                      destPort == flow[1], protocol == flow[4]]):
                flow[7] += 1
                flow[8] += numBytes
                flow[9] += 1
                flow[10] += numBytes
                flow[12] = timeReceived - flow[11]
                isNewFlow = False

        # If flow isn't in list, create new flow and append it
        if (isNewFlow):
            newFlow = [srceIP, srcePort, destIP, destPort, protocol, 1,
                       numBytes, 0, 0, 1, numBytes, timeReceived, timeReceived,
                       flowLabel]
            flowList.append(newFlow)

        # print x.summary()

        # x.show()


def outputToFlowCSV(flowList):
    '''
    Create a CSV file given a list of flows, or append to the file if it already
    exists.
    '''

    # Open or create a CSV file to write flows to
    with open('data.csv', mode='a+') as flowsCSV:
        # Creater writer to write flows to file
        flowWriter = csv.writer(flowsCSV, delimiter=',')

        # Write flow-by-flow to file
        for flow in flowList:
            flowWriter.writerow(flow)


def trimFlowList(flowList):
    '''
    Eliminate noise and useless flows by returning only flows that consist of
    10 packets or more.
    '''
    trimmedFlowList = []
    for flow in flowList:
        if (flow[9] >= 10):
            trimmedFlowList.append(flow)
    return trimmedFlowList


def main():

    # Labels for different packet scenarios
    labelList = ["Web Browsing",
                 "Video Streaming",
                 "Video Conferencing",
                 "File Downloading"]

    # Initial empty flow list
    flowList = []

    for label in labelList:
        # Before sniffing, prompt user to begin activity for given scenario
        input("Start sniffing for " + label + " data?")

        # Repeat sniffing until at least 25 samples per scenario
        while (len(flowList) <= 25):
            print("Continue activity..")
            pkts = sniff(prn = lambda x: packetExtraction(x, flowList, labelList.index(label) + 1), count = 2000)

            # Eliminate packets noise and useless flows
            flowList = trimFlowList(flowList)

        # Output flows for the given scenario to CSV and then reset flowList for
        # next scenario
        outputToFlowCSV(flowList)
        flowList = []

main()
