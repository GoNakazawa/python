# -*- coding: utf-8 -*-
"""
Created on Sun Aug 15 10:58:31 2021

@author: 13191
"""
SELECT Album.Title, Artist.Name FROM Album
   ...> INNER JOIN Artist ON Album.ArtistId = Artist.ArtistId LIMIT 10;
   
   
   SELECT c.FirstName as CustomerName,
   e.FirstName AS SupportRepName,
   e.Title AS SupportRepTitle
   
   FROM Customer c
   
   LEFT JOIN Employee e ON c.SupportRepId = e.EmployeeId;
    
