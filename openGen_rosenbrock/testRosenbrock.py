#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 09:31:38 2023

@author: kst
"""

import opengen as og


# Use TCP server
# ------------------------------------
mng = og.tcp.OptimizerTcpManager('my_optimizers/rosenbrock')
mng.start()

mng.ping()
server_response = mng.call([1.0, 50.0])

if server_response.is_ok():
    solution = server_response.get()
    u_star = solution.solution
    status = solution.exit_status

mng.kill()